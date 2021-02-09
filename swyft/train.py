# pylint: disable=no-member, not-callable
from contextlib import suppress
from copy import deepcopy

import numpy as np
import torch

from .types import Array, Combinations, Dict, Sequence, Union
from .utils import combine_z, dict_to_device, dict_to_tensor, verbosity


def double_features(f):
    """Double feature vector as (A, B, C, D) --> (A, A, B, B, C, C, D, D)

    Args:
        f (tensor): Feature vectors (n_batch, n_features)
    Returns:
        f (tensor): Feature vectors (2*n_btach. n_features)
    """
    return torch.repeat_interleave(f, 2, dim=0)


def double_params(params):
    """Double parameters as (A, B, C, D) --> (A, B, A, B, C, D, C, D) etc

    Args:
        params (dict): Dictionary of parameters with shape (n_batch).
    Returns:
        dict: Dictionary of parameters with shape (2*n_batch).
    """
    out = {}
    for k, v in params.items():
        out[k] = torch.repeat_interleave(v.view(-1, 2), 2, dim=0).flatten()
    return out


def loss_fn(head, tail, obs, params):
    # Get features
    f = head(obs)
    n_batch = f.shape[0]

    assert n_batch % 2 == 0, "Loss function can only handle even-numbered batch sizes."

    # Repeat interleave
    f_doubled = double_features(f)
    params_doubled = double_params(params)

    # Get
    lnL = tail(f_doubled, params_doubled)
    lnL = lnL.view(-1, 4, lnL.shape[-1])

    loss = -torch.nn.functional.logsigmoid(lnL[:, 0])
    loss += -torch.nn.functional.logsigmoid(-lnL[:, 1])
    loss += -torch.nn.functional.logsigmoid(-lnL[:, 2])
    loss += -torch.nn.functional.logsigmoid(lnL[:, 3])
    loss = loss.sum(axis=0) / n_batch

    return loss


def split_length_by_percentage(length: int, percents: Sequence[float]) -> Sequence[int]:
    assert np.isclose(sum(percents), 1.0), f"{percents} does not sum to 1."
    lengths = [int(percent * length) for percent in percents]

    # Any extra from round off goes to the first split.
    difference = length - sum(lengths)
    lengths[0] += difference
    assert length == sum(
        lengths
    ), f"Splitting into {lengths} should equal total {length}."
    return lengths


# We have the posterior exactly because our proir is known and flat. Flip bayes theorem, we have the likelihood ratio.
# Consider that the variance of the loss from different legs causes some losses to have high coefficients in front of them.
def train(
    head,
    tail,
    train_loader,
    validation_loader,
    early_stopping_patience,
    max_epochs=None,
    lr=1e-3,
    combinations=None,
    device="cpu",
    non_blocking=True,
):
    """Network training loop.

    Args:
        network (nn.Module): network for ratio estimation.
        train_loader (DataLoader): DataLoader of samples.
        validation_loader (DataLoader): DataLoader of samples.
        max_epochs (int): Number of epochs.
        lr (float): learning rate.
        combinations (list, optional): determines posteriors that are generated.
            examples:
                [[0,1], [3,4]]: p(z_0,z_1) and p(z_3,z_4) are generated
                    initialize network with zdim = 2, pdim = 2
                [[0,1,5,2]]: p(z_0,z_1,z_5,z_2) is generated
                    initialize network with zdim = 1, pdim = 4
        device (str, device): Move batches to this device.
        non_blocking (bool): non_blocking in .to(device) expression.

    Returns:
        list: list of training losses.
    """
    # TODO consider that the user might want other training stats, like number of correct samples for example
    def do_epoch(loader: torch.utils.data.dataloader.DataLoader, train: bool):
        accumulated_loss = 0
        training_context = suppress() if train else torch.no_grad()
        with training_context:
            for batch in loader:
                optimizer.zero_grad()

                obs = dict_to_device(
                    batch["obs"], device=device, non_blocking=non_blocking
                )
                params = dict_to_device(
                    batch["par"], device=device, non_blocking=non_blocking
                )
                loss = loss_fn(head, tail, obs, params)
                loss = sum(loss)

                if train:
                    loss.backward()
                    optimizer.step()

                accumulated_loss += loss.detach().cpu().numpy().item()

        return accumulated_loss

    max_epochs = 2 ** 31 - 1 if max_epochs is None else max_epochs
    params = list(head.parameters()) + list(tail.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    n_train_batches = len(train_loader) if len(train_loader) != 0 else 1
    n_validation_batches = len(validation_loader) if len(validation_loader) != 0 else 1

    train_losses, validation_losses = [], []
    epoch, fruitless_epoch, min_loss = 0, 0, float("Inf")
    while epoch < max_epochs and fruitless_epoch < early_stopping_patience:
        head.train()
        tail.train()
        train_loss = do_epoch(train_loader, True)
        train_losses.append(train_loss / n_train_batches)

        # network.eval()
        head.eval()
        tail.eval()
        validation_loss = do_epoch(validation_loader, False)
        if verbosity() >= 2:
            print("  val loss = %.4g" % (validation_loss / n_validation_batches))
        validation_losses.append(validation_loss / n_validation_batches)

        epoch += 1
        if epoch == 0 or min_loss > validation_loss:
            fruitless_epoch = 0
            min_loss = validation_loss
            # TODO update
            best_state_dict_head = deepcopy(head.state_dict())
            best_state_dict_tail = deepcopy(tail.state_dict())
        else:
            fruitless_epoch += 1

    return train_losses, validation_losses, best_state_dict_head, best_state_dict_tail


def trainloop(
    head,
    tail,
    dataset,
    combinations=None,
    batch_size=32,
    nworkers=4,
    max_epochs=50,
    early_stopping_patience=1,
    device="cpu",
    lr_schedule=[1e-3, 3e-4, 1e-4],
    percent_validation=0.1,
):
    percent_train = 1.0 - percent_validation
    ntrain, nvalid = split_length_by_percentage(
        len(dataset), (percent_train, percent_validation)
    )
    dataset_train, dataset_valid = torch.utils.data.random_split(
        dataset, [ntrain, nvalid]
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=nworkers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=batch_size,
        num_workers=nworkers,
        pin_memory=True,
        drop_last=True,
    )

    # Train!
    train_loss, valid_loss = [], []
    for i, lr in enumerate(lr_schedule):
        if verbosity() >= 2:
            print("  lr = %.4g" % lr)
        tl, vl, sd_head, sd_tail = train(
            head,
            tail,
            train_loader,
            valid_loader,
            early_stopping_patience=early_stopping_patience,
            lr=lr,
            max_epochs=max_epochs,
            device=device,
            combinations=combinations,
        )
        vl_minimum = min(vl)
        vl_min_idx = vl.index(vl_minimum)
        train_loss.append(tl[: vl_min_idx + 1])
        valid_loss.append(vl[: vl_min_idx + 1])
        head.load_state_dict(sd_head)
        tail.load_state_dict(sd_tail)


# def get_statistics(
#    points: Union["swyft.estimation.Points", Sequence[Dict[str, Array]]],
#    combinations: Combinations = None,
#    n_samples: int = 300,
# ):
#    """Calculate the mean and std of both x and z.
#
#    Args:
#        points: list of dictionaries with keys 'x' and 'z'.
#        combinations
#        n_samples: size of sample
#
#    Returns:
#        x_mean, x_std, z_mean, z_std
#    """
#    irand = np.random.choice(len(points), n_samples)
#    x = [points[i]["x"] for i in irand]
#    z = [points[i]["z"] for i in irand]
#    x_mean = sum(x) / len(x)
#    z_mean = sum(z) / len(z)
#    x_var = sum([(x[i] - x_mean) ** 2 for i in range(len(x))]) / len(x)
#    z_var = sum([(z[i] - z_mean) ** 2 for i in range(len(z))]) / len(z)
#
#    z_mean = combine_z(z_mean, combinations)
#    z_var = combine_z(z_var, combinations)
#
#    return x_mean, x_var ** 0.5, z_mean, z_var ** 0.5


if __name__ == "__main__":
    pass
