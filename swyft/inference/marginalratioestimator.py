from typing import Callable, Dict, Hashable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
from toolz.dicttoolz import valmap
from torch.utils.data import DataLoader, Dataset, random_split

import swyft
import swyft.utils
from swyft.inference.train import get_ntrain_nvalid
from swyft.types import Array, Device, MarginalIndex, ObsType, RatioType
from swyft.utils.array import array_to_tensor, dict_array_to_tensor
from swyft.utils.parameters import tupleize_marginals

SchedulerType = Union[
    torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
]


def split_length_by_percentage(length: int, percents: Sequence[float]) -> Sequence[int]:
    """Given the length of a sequence, return the indices which would divide it into `percents` parts.
    Any rounding errors go into the first part.

    Args:
        length
        percents

    Returns:
        length_of_parts
    """
    assert np.isclose(sum(percents), 1.0), f"{percents} does not sum to 1."
    lengths = [int(percent * length) for percent in percents]

    # Any extra from round off goes to the first split.
    difference = length - sum(lengths)
    lengths[0] += difference
    assert length == sum(
        lengths
    ), f"Splitting into {lengths} should equal total {length}."
    return lengths


def get_ntrain_nvalid(
    validation_amount: Union[float, int], len_dataset: int
) -> Tuple[int, int]:
    """Divide a dataset into a training and validation set.

    Args:
        validation_amount: percentage or number of elements in the validation set
        len_dataset: total length of the dataset

    Raises:
        TypeError: When the validation_amount is neither a float or int.

    Returns:
        (n_train, n_valid)
    """
    assert validation_amount > 0
    if isinstance(validation_amount, float):
        percent_validation = validation_amount
        percent_train = 1.0 - percent_validation
        n_valid, n_train = split_length_by_percentage(
            len_dataset, (percent_validation, percent_train)
        )
        if n_valid % 2 != 0:
            n_valid += 1
            n_train -= 1
    elif isinstance(validation_amount, int):
        n_valid = validation_amount
        n_train = len_dataset - n_valid
        assert n_train > 0

        if n_valid % 2 != 0:
            n_valid += 1
            n_train -= 1
    else:
        raise TypeError()
    return n_train, n_valid


def double_features(f: torch.Tensor) -> torch.Tensor:
    """Double feature vector as (A, B, C, D) --> (A, A, B, B, C, C, D, D)

    Args:
        f: Feature vectors (n_batch, n_features)

    Returns:
        f: Feature vectors (2*n_btach. n_features)
    """
    return torch.repeat_interleave(f, 2, dim=0)


def double_params(params: torch.Tensor) -> torch.Tensor:
    """Double parameters as (A, B, C, D) --> (A, B, A, B, C, D, C, D) etc

    Args:
        params: Dictionary of parameters with shape (n_batch).

    Returns:
        dict: Dictionary of parameters with shape (2*n_batch).
    """
    n = params.shape[-1]
    out = torch.repeat_interleave(params.view(-1, 2 * n), 2, dim=0).view(-1, n)
    return out


MarginalRatioEstimatorType = TypeVar(
    "MarginalRatioEstimatorType", bound="MarginalRatioEstimator"
)


# TODO change the ratio estimator to train / evaluate on v
class MarginalRatioEstimator:
    def __init__(
        self,
        marginal_indices: MarginalIndex,
        network: nn.Module,
        device: Device,
    ) -> None:
        self.marginal_indices = tupleize_marginals(marginal_indices)
        self.device = device
        self.network = network
        self.network.to(self.device)

        self.epoch = None
        self.best_network_state_dict = None
        self.min_loss = float("-Inf")
        self.optimizer = None
        self.scheduler = None

    def train(
        self,
        dataset: Dataset,
        batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_percentage: float = 0.1,
        optimizer: Callable = torch.optim.Adam,
        scheduler: Optional[Callable] = torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs: dict = {"factor": 0.1, "patience": 5},
        early_stopping_patience: Optional[int] = 25,
        max_epochs: int = 2 ** 31 - 1,
        nworkers: int = 0,
        non_blocking: bool = True,
        pin_memory: bool = True,
    ) -> None:
        if early_stopping_patience is None:
            early_stopping_patience = max_epochs

        if self.optimizer is None:
            self.optimizer = optimizer(self.network.parameters(), lr=learning_rate)

        if scheduler is not None and self.scheduler is None:
            self.scheduler = scheduler(self.optimizer, **scheduler_kwargs)

        n_train, n_valid = get_ntrain_nvalid(validation_percentage, len(dataset))
        dataset_train, dataset_valid = random_split(dataset, [n_train, n_valid])
        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            num_workers=nworkers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        valid_loader = DataLoader(
            dataset_valid,
            batch_size=min(batch_size, n_valid),
            num_workers=nworkers,
            pin_memory=pin_memory,
            drop_last=True,
        )

        n_validation_batches = len(valid_loader) if len(valid_loader) != 0 else 1
        validation_losses = []
        self.epoch, fruitless_epoch = 0, 0
        while self.epoch < max_epochs and fruitless_epoch < early_stopping_patience:
            # Training
            self.network.train()
            for observation, u, _ in train_loader:
                self.optimizer.zero_grad()
                observation = swyft.utils.dict_to_device(
                    observation, device=self.device, non_blocking=non_blocking
                )
                u = u.to(self.device)
                loss = self._loss(observation, u).sum(dim=0)
                loss.backward()
                self.optimizer.step()

            self.epoch += 1

            # Evaluation
            self.network.eval()
            loss_sum = 0
            with torch.inference_mode():
                for observation, u, _ in valid_loader:
                    observation = swyft.utils.dict_to_device(
                        observation, device=self.device, non_blocking=non_blocking
                    )
                    u = u.to(self.device)
                    validation_loss = self._loss(observation, u).sum(dim=0)
                    loss_sum += validation_loss
                loss_avg = loss_sum / n_validation_batches
                print(
                    "\rtraining: lr=%.2g, epoch=%i, validation loss=%.4g"
                    % (self._get_last_lr(self.scheduler), self.epoch, loss_avg),
                    end="",
                    flush=True,
                )
                validation_losses.append(loss_avg)

                if self.epoch == 0 or self.min_loss > validation_loss:
                    fruitless_epoch = 0
                    self.min_loss = validation_loss
                    self.best_network_state_dict = self.network.state_dict()
                else:
                    fruitless_epoch += 1

                if self.scheduler is not None:
                    self.scheduler.step(loss_avg)

    def log_ratio(
        self,
        observation: ObsType,
        parameters: Array,
        n_batch: Optional[int] = None,
        inference_mode: bool = True,
    ) -> RatioType:
        was_training = self.network.training
        self.network.eval()

        context = torch.inference_mode if inference_mode else torch.no_grad
        with context:
            observation = dict_array_to_tensor(observation, device=self.device)
            observation = valmap(lambda x: x.unsqueeze(0), observation)
            len_parameters = len(parameters)
            if n_batch is None or len_parameters <= n_batch:
                parameters = array_to_tensor(parameters, device=self.device)
                ratio = self.network(observation, parameters).cpu().numpy()
            else:
                ratio = []
                for i in range(len_parameters // n_batch + 1):
                    parameter_batch = array_to_tensor(
                        parameters[i * n_batch : (i + 1) * n_batch, :],
                        device=self.device,
                    )
                    ratio_batch = (
                        self.network(observation, parameter_batch).cpu().numpy()
                    )
                    ratio.append(ratio_batch)
                ratio = np.vstack(ratio)

        if was_training:
            self.network.train()
        else:
            self.network.eval()

        return {k: ratio[..., i] for i, k in enumerate(self.marginal_indices)}

    @staticmethod
    def _get_last_lr(scheduler: SchedulerType) -> float:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if scheduler.best == float("Inf"):
                return scheduler.optimizer.param_groups[0]["lr"]
            else:
                return scheduler._last_lr[-1]
        elif isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            return scheduler.get_last_lr()
        else:
            raise NotImplementedError(
                f"Cannot determine learning_rate from {scheduler}"
            )

    def _loss(
        self, observation: Dict[Hashable, torch.Tensor], parameters: torch.Tensor
    ) -> torch.Tensor:
        """Returns marginal-wise losses."""
        n_batch = parameters.size(0)
        assert (
            n_batch % 2 == 0
        ), "Loss function can only handle even-numbered batch sizes."
        assert all(
            [v.size(0) == n_batch for v in observation.values()]
        ), "The observation batch_size must agree with the parameter batch_size."

        # Repeat interleave
        observation_doubled = valmap(double_features, observation)
        parameters_doubled = double_params(parameters)

        lnL = self.network(observation_doubled, parameters_doubled)
        lnL = lnL.view(-1, 4, lnL.shape[-1])

        loss = -torch.nn.functional.logsigmoid(lnL[:, 0])
        loss += -torch.nn.functional.logsigmoid(-lnL[:, 1])
        loss += -torch.nn.functional.logsigmoid(-lnL[:, 2])
        loss += -torch.nn.functional.logsigmoid(lnL[:, 3])
        loss = loss.sum(axis=0) / n_batch

        return loss

    def state_dict(self) -> dict:
        return {
            "marginal_indices": self.marginal_indices,
            "network": self.network.state_dict(),
            "epoch": self.epoch,
            "min_loss": self.min_loss,
            "best_network_state_dict": self.best_network_state_dict,
            "optimizer": self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }  # TODO this could also save the __class__ and use importlib, thereby reducing the arguments to from_state_dict

    @classmethod
    def from_state_dict(
        cls,
        network: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[SchedulerType],
        device: Device,
        state_dict: dict,
    ) -> MarginalRatioEstimatorType:
        marginal_ratio_estimator = cls.__new__(cls)

        marginal_ratio_estimator.marginal_indices = state_dict["marginal_indices"]
        marginal_ratio_estimator.epoch = state_dict["epoch"]
        marginal_ratio_estimator.min_loss = state_dict["min_loss"]
        marginal_ratio_estimator.best_network_state_dict = state_dict[
            "best_network_state_dict"
        ]
        marginal_ratio_estimator.device = device

        marginal_ratio_estimator.network = network
        marginal_ratio_estimator.optimizer = optimizer
        marginal_ratio_estimator.scheduler = scheduler

        marginal_ratio_estimator.network.load_state_dict(state_dict["network"])
        marginal_ratio_estimator.optimizer.load_state_dict(state_dict["optimizer"])
        marginal_ratio_estimator.scheduler.load_state_dict(state_dict["scheduler"])
        return marginal_ratio_estimator


if __name__ == "__main__":
    pass
