# pylint: disable=no-member, not-callable
import logging
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np
import torch

from swyft.inference.loss import loss_fn
from swyft.types import Device
from swyft.utils import dict_to_device

log = logging.getLogger(__name__)


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


@dataclass
class TrainOptions:
    """Settings for the trainloop function. Defaults are specified in swyft.Posteriors.train."""

    batch_size: int  # = 64
    validation_size: float  # = 0.1
    early_stopping_patience: int  # = 10
    max_epochs: int  # = 50
    optimizer: Callable[..., "torch.optim.Optimizer"]  # = torch.optim.Adam
    optimizer_args: dict  # = field(default_factory=lambda: dict(lr=1e-3))
    scheduler: Callable[
        ..., "torch.optim.lr_scheduler._LRScheduler"
    ]  # = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_args: dict  # = field(default_factory=lambda: dict(factor=0.1, patience=5))
    nworkers: int  # = 2
    non_blocking: bool  # = True


# We have the posterior exactly because our prior is known and flat. Flip bayes theorem, we have the likelihood ratio.
# Consider that the variance of the loss from different legs causes some losses to have high coefficients in front of them.
def do_training(
    head: torch.nn.Module,
    tail: torch.nn.Module,
    train_loader: torch.utils.data.dataloader.DataLoader,
    validation_loader: torch.utils.data.dataloader.DataLoader,
    trainoptions: TrainOptions,
    device: Device,
) -> Tuple:
    """Network training loop.

    Args:
        head:
        tail:
        train_loader:
        validation_loader:
        trainoptions:
        device:

    Returns:
        train_losses, validation_losses, best_state_dict_head, best_state_dict_tail
    """
    # TODO consider that the user might want other training stats, like number of correct samples for example
    def do_epoch(loader: torch.utils.data.dataloader.DataLoader, train: bool):
        accumulated_loss = 0
        training_context = suppress() if train else torch.no_grad()
        with training_context:
            for batch in loader:
                optimizer.zero_grad()
                sim, z, _ = batch

                obs = dict_to_device(
                    sim, device=device, non_blocking=trainoptions.non_blocking
                )
                params = z.to(device=device, non_blocking=trainoptions.non_blocking)
                losses = loss_fn(head, tail, obs, params)
                loss = sum(losses)

                if train:
                    loss.backward()
                    optimizer.step()

                accumulated_loss += loss.detach().cpu().numpy().item()

        return accumulated_loss

    max_epochs = (
        2 ** 31 - 1 if trainoptions.max_epochs is None else trainoptions.max_epochs
    )
    params = list(head.parameters()) + list(tail.parameters())
    optimizer = trainoptions.optimizer(params, **trainoptions.optimizer_args)
    scheduler = trainoptions.scheduler(optimizer, **trainoptions.scheduler_args)

    n_train_batches = len(train_loader) if len(train_loader) != 0 else 1
    n_validation_batches = len(validation_loader) if len(validation_loader) != 0 else 1

    train_losses, validation_losses = [], []
    epoch, fruitless_epoch, min_loss = 0, 0, float("Inf")
    while epoch < max_epochs and fruitless_epoch < trainoptions.early_stopping_patience:
        head.train()
        tail.train()
        train_loss = do_epoch(train_loader, True)
        train_losses.append(train_loss / n_train_batches)

        # network.eval()
        head.eval()
        tail.eval()
        validation_loss = do_epoch(validation_loader, False)
        l = validation_loss / n_validation_batches
        logging.debug("validation loss = %.4g" % l)
        epoch += 1
        # FIXME: Not optimal when multiple parameter groups are present
        lr = optimizer.param_groups[0]["lr"]
        print(
            "\rTraining: lr=%.2g, Epoch=%i, VL=%.4g" % (lr, epoch, l),
            end="",
            flush=True,
        )
        validation_losses.append(l)

        if epoch == 0 or min_loss > validation_loss:
            fruitless_epoch = 0
            min_loss = validation_loss
            # TODO update
            best_state_dict_head = deepcopy(head.state_dict())
            best_state_dict_tail = deepcopy(tail.state_dict())
        else:
            fruitless_epoch += 1
        scheduler.step(l)

    print("")

    return train_losses, validation_losses, best_state_dict_head, best_state_dict_tail


def _get_ntrain_nvalid(validation_size, len_dataset):
    if isinstance(validation_size, float):
        percent_validation = validation_size
        percent_train = 1.0 - percent_validation
        ntrain, nvalid = split_length_by_percentage(
            len_dataset, (percent_train, percent_validation)
        )
        if nvalid % 2 != 0:
            nvalid += 1
            ntrain -= 1
    elif isinstance(validation_size, int):
        nvalid = validation_size
        ntrain = len_dataset - nvalid
        assert ntrain > 0

        if nvalid % 2 != 0:
            nvalid += 1
            ntrain -= 1
    else:
        raise TypeError()
    return ntrain, nvalid


def trainloop(
    head: torch.nn.Module,
    tail: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    trainoptions: TrainOptions,
    device: Device = "cpu",
) -> dict:
    log.debug("Entering trainloop")
    log.debug(f"{'batch_size':>25} {trainoptions.batch_size:<4}")
    log.debug(f"{'validation_size':>25} {trainoptions.validation_size:<4}")
    log.debug(
        f"{'early_stopping_patience':>25} {trainoptions.early_stopping_patience:<4}"
    )
    log.debug(f"{'max_epochs':>25} {trainoptions.max_epochs:<4}")
    log.debug(f"{'optimizer_fn':>25} {repr(trainoptions.optimizer):<4}")
    log.debug(f"{'scheduler_fn':>25} {repr(trainoptions.scheduler):<4}")
    log.debug(f"{'nworkers':>25} {trainoptions.nworkers:<4}")

    assert trainoptions.validation_size > 0
    ntrain, nvalid = _get_ntrain_nvalid(trainoptions.validation_size, len(dataset))

    dataset_train, dataset_valid = torch.utils.data.random_split(
        dataset, [ntrain, nvalid]
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=trainoptions.batch_size,
        num_workers=trainoptions.nworkers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=min(trainoptions.batch_size, nvalid),
        num_workers=trainoptions.nworkers,
        pin_memory=True,
        drop_last=True,
    )
    tl, vl, sd_head, sd_tail = do_training(
        head, tail, train_loader, valid_loader, trainoptions, device
    )
    vl_minimum = min(vl)
    vl_min_idx = vl.index(vl_minimum)
    train_loss = tl[: vl_min_idx + 1]
    valid_loss = vl[: vl_min_idx + 1]
    head.load_state_dict(sd_head)
    tail.load_state_dict(sd_tail)
    log.debug("Train losses: " + str(train_loss))
    log.debug("Valid losses: " + str(valid_loss))
    log.debug("Finished trainloop.")
    return dict(train_loss=train_loss, valid_loss=valid_loss)


if __name__ == "__main__":
    pass
