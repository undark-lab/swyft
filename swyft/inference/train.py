# pylint: disable=no-member, not-callable
import logging
from contextlib import suppress
from copy import deepcopy
from typing import Sequence

import numpy as np
import torch

from swyft.utils import dict_to_device

from .loss import loss_fn


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
    max_epochs,
    lr,
    reduce_lr_factor,
    reduce_lr_patience,
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
                sim, z = batch

                obs = dict_to_device(sim, device=device, non_blocking=non_blocking)
                params = z.to(device, non_blocking=non_blocking)
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
    )

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
        avg_validation_loss = validation_loss / n_validation_batches
        logging.debug("validation loss = %.4g" % (avg_validation_loss))
        validation_losses.append(avg_validation_loss)

        epoch += 1
        if epoch == 0 or min_loss > validation_loss:
            fruitless_epoch = 0
            min_loss = validation_loss
            # TODO update
            best_state_dict_head = deepcopy(head.state_dict())
            best_state_dict_tail = deepcopy(tail.state_dict())
        else:
            fruitless_epoch += 1
        scheduler.step(avg_validation_loss)

    return train_losses, validation_losses, best_state_dict_head, best_state_dict_tail


def trainloop(
    head,
    tail,
    dataset,
    batch_size=64,
    percent_validation=0.1,
    early_stopping_patience=10,
    max_epochs=50,
    lr=1e-3,
    reduce_lr_factor=0.1,
    reduce_lr_patience=5,
    nworkers=0,
    device="cpu",
):
    logging.debug("Entering trainloop")
    logging.debug(f"  batch_size {batch_size:>5}")
    logging.debug(f"  percent_validation {percent_validation:>5}")
    logging.debug(f"  early_stopping_patience {early_stopping_patience:>5}")
    logging.debug(f"  max_epochs {max_epochs:>5}")
    logging.debug(f"  lr {lr:>5}")
    logging.debug(f"  reduce_lr_factor {reduce_lr_factor:>5}")
    logging.debug(f"  reduce_lr_patience {reduce_lr_patience:>5}")
    logging.debug(f"  nworkers {nworkers:>5}")

    percent_train = 1.0 - percent_validation
    nvalid, ntrain = split_length_by_percentage(
        len(dataset), (percent_validation, percent_train)
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

    tl, vl, sd_head, sd_tail = train(
        head,
        tail,
        train_loader,
        valid_loader,
        early_stopping_patience=early_stopping_patience,
        max_epochs=max_epochs,
        lr=lr,
        reduce_lr_factor=reduce_lr_factor,
        reduce_lr_patience=reduce_lr_patience,
        device=device,
    )
    vl_minimum = min(vl)
    vl_min_idx = vl.index(vl_minimum)
    train_loss = tl[: vl_min_idx + 1]
    valid_loss = vl[: vl_min_idx + 1]
    head.load_state_dict(sd_head)
    tail.load_state_dict(sd_tail)
    logging.debug("Train losses: " + str(train_loss))
    logging.debug("Valid losses: " + str(valid_loss))
    logging.debug("Finished trainloop.")
    return dict(train_loss=train_loss, valid_loss=valid_loss)


if __name__ == "__main__":
    pass
