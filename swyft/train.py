# pylint: disable=no-member, not-callable
from copy import deepcopy
from contextlib import suppress

import numpy as np
import torch

from .utils import combine_z
from .types import Sequence


def loss_fn(network, xz, combinations=None):
    """Evaluate binary-cross-entropy loss function. Mean over batch.

    Args:
        network (nn.Module): network taking minibatch of samples and returing ratio estimator.
        xz (dict): batch of samples to train on.
        combinations (list, optional): determines posteriors that are generated.
            examples:
                [[0,1], [3,4]]: p(z_0,z_1) and p(z_3,z_4) are generated
                    initialize network with zdim = 2, pdim = 2
                [[0,1,5,2]]: p(z_0,z_1,z_5,z_2) is generated
                    initialize network with zdim = 1, pdim = 4

    Returns:
        Tensor: training loss.
    """
    assert xz["x"].size(0) == xz["z"].size(0), "Number of x and z must be equal."
    assert (
        xz["x"].size(0) % 2 == 0
    ), "There must be an even number of samples in the batch for contrastive learning."
    n_batch = xz["x"].size(0)

    # Is it the removal of replacement that made it stop working?!

    # bring x into shape
    # (n_batch*2, data-shape)  - repeat twice each sample of x - there are n_batch samples
    # repetition pattern in first dimension is: [a, a, b, b, c, c, d, d, ...]
    x = xz["x"]
    x = torch.repeat_interleave(x, 2, dim=0)

    # bring z into shape
    # (n_batch*2, param-shape)  - repeat twice each sample of z - there are n_batch samples
    # repetition is alternating in first dimension: [a, b, a, b, c, d, c, d, ...]
    z = xz["z"]
    z = torch.stack([combine_z(zs, combinations) for zs in z])
    zdim = len(z[0])
    z = z.view(n_batch // 2, -1, *z.shape[-1:])
    z = torch.repeat_interleave(z, 2, dim=0)
    z = z.view(n_batch * 2, -1, *z.shape[-1:])

    # call network
    lnL = network(x, z)
    lnL = lnL.view(n_batch // 2, 4, zdim)

    # Evaluate cross-entropy loss
    # loss =
    # -ln( exp(lnL(x_a, z_a))/(1+exp(lnL(x_a, z_a))) )
    # -ln( exp(lnL(x_b, z_b))/(1+exp(lnL(x_b, z_b))) )
    # -ln( 1/(1+exp(lnL(x_a, z_b))) )
    # -ln( 1/(1+exp(lnL(x_b, z_a))) )
    loss = -torch.nn.functional.logsigmoid(lnL[:, 0])
    loss += -torch.nn.functional.logsigmoid(-lnL[:, 1])
    loss += -torch.nn.functional.logsigmoid(-lnL[:, 2])
    loss += -torch.nn.functional.logsigmoid(lnL[:, 3])
    loss = loss.sum() / (n_batch // 2)

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
    network,
    train_loader,
    validation_loader,
    early_stopping_patience,
    max_epochs=None,
    lr=1e-3,
    combinations=None,
    device=None,
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
                if device is not None:
                    batch = {
                        k: v.to(device, non_blocking=non_blocking)
                        for k, v in batch.items()
                    }
                loss = loss_fn(network, batch, combinations=combinations)
                if train:
                    loss.backward()
                    optimizer.step()
                accumulated_loss += loss.detach().cpu().numpy().item()
        return accumulated_loss

    max_epochs = 2 ** 31 - 1 if max_epochs is None else max_epochs
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    n_train_batches = len(train_loader) if len(train_loader) != 0 else 1
    n_validation_batches = len(validation_loader) if len(validation_loader) != 0 else 1

    train_losses, validation_losses = [], []
    epoch, fruitless_epoch, min_loss = 0, 0, float("Inf")
    while epoch < max_epochs and fruitless_epoch < early_stopping_patience:
        # print("Epoch:", epoch, end = "\r")
        network.train()
        train_loss = do_epoch(train_loader, True)
        train_losses.append(train_loss / n_train_batches)

        network.eval()
        validation_loss = do_epoch(validation_loader, False)
        print("Validation loss:", validation_loss)
        validation_losses.append(validation_loss / n_validation_batches)

        epoch += 1
        if epoch == 0 or min_loss > validation_loss:
            fruitless_epoch = 0
            min_loss = validation_loss
            best_state_dict = deepcopy(network.state_dict())
        else:
            fruitless_epoch += 1

    print("Total epochs:", epoch)
    # print("Validation losses:", validation_losses)
    return train_losses, validation_losses, best_state_dict


def trainloop(
    net,
    dataset,
    combinations=None,
    batch_size=32,
    nworkers=4,
    max_epochs=50,
    early_stopping_patience=1,
    device="cpu",
    lr_schedule=[1e-3, 1e-4, 1e-5],
    percent_validation=0.1,
):
    print("Start training")
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
        print(f"LR iteration {i}")
        tl, vl, sd = train(
            net,
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
        net.load_state_dict(sd)


def get_norms(xz, combinations=None, N=300):
    irand = np.random.choice(len(xz), N)
    x = [xz[i]["x"] for i in irand]
    z = [xz[i]["z"] for i in irand]
    x_mean = sum(x) / len(x)
    z_mean = sum(z) / len(z)
    x_var = sum([(x[i] - x_mean) ** 2 for i in range(len(x))]) / len(x)
    z_var = sum([(z[i] - z_mean) ** 2 for i in range(len(z))]) / len(z)

    z_mean = combine_z(z_mean, combinations)
    z_var = combine_z(z_var, combinations)

    # print("Normalizations")
    # print("x_mean", x_mean)
    # print("x_err", x_var**0.5)
    # print("z_mean", z_mean)
    # print("z_err", z_var**0.5)

    return x_mean, x_var ** 0.5, z_mean, z_var ** 0.5


if __name__ == "__main__":
    pass
