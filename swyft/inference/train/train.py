# pylint: disable=no-member, not-callable
import logging
from contextlib import suppress
from copy import deepcopy
from typing import Sequence

import numpy as np
import torch

from swyft.inference.train.loss import loss_fn
from swyft.utils import dict_to_device


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
                losses = loss_fn(head, tail, obs, params)
                loss = sum(losses)

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
        logging.debug(
            "validation loss = %.4g" % (validation_loss / n_validation_batches)
        )
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
    logging.debug("Entering trainloop")
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
    for _, lr in enumerate(lr_schedule):
        logging.debug("lr: %.3g" % lr)
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
        train_loss += tl[: vl_min_idx + 1]
        valid_loss += vl[: vl_min_idx + 1]
        head.load_state_dict(sd_head)
        tail.load_state_dict(sd_tail)
    logging.debug("Train losses: " + str(train_loss))
    logging.debug("Valid losses: " + str(valid_loss))
    logging.debug("Finished trainloop.")
    return dict(train_loss=train_loss, valid_loss=valid_loss)


if __name__ == "__main__":
    pass
