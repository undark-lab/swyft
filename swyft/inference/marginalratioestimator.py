from typing import (
    Callable,
    Dict,
    Hashable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from toolz.dicttoolz import valmap
from torch.utils.data import DataLoader, Dataset, random_split

import swyft
import swyft.utils
from swyft.saveable import StateDictSaveable, StateDictSaveableType
from swyft.types import Array, Device, MarginalIndex, MarginalToArray, ObsType, PathType
from swyft.utils.array import array_to_tensor, dict_array_to_tensor
from swyft.utils.marginals import tupleize_marginal_indices

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
        raise TypeError("validation_amount must be int or float")
    return n_train, n_valid


def double_observation(f: torch.Tensor) -> torch.Tensor:
    """Double observation vector as (A, B, C, D) --> (A, A, B, B, C, C, D, D)

    Args:
        f: Observation vectors (n_batch, ...)

    Returns:
        Observation vectors (2*n_batch, ...)
    """
    return torch.repeat_interleave(f, 2, dim=0)


def double_parameters(parameters: torch.Tensor) -> torch.Tensor:
    """Double parameters as (A, B, C, D) --> (A, B, A, B, C, D, C, D) etc

    Args:
        parameters: Parameter vectors (n_batch, n_parameters).

    Returns:
        parameters with shape (2*n_batch, n_parameters).
    """
    n_batch, n_parameters = parameters.shape
    assert n_batch % 2 == 0, "n_batch must be divisible by two."
    out = torch.repeat_interleave(parameters.view(-1, 2 * n_parameters), 2, dim=0).view(
        -1, n_parameters
    )
    return out


MarginalRatioEstimatorType = TypeVar(
    "MarginalRatioEstimatorType", bound="MarginalRatioEstimator"
)


class MarginalRatioEstimator(StateDictSaveable):
    """Handles the training and evaluation of a ratio estimator. Which ratios are defined by the `marginal_indices` attribute.
    The network must take observation dictionaries and parameter arrays and produce estimated an `log_ratio` for every marginal of interest.
    """

    def __init__(
        self,
        marginal_indices: MarginalIndex,
        network: nn.Module,
        device: Device,
    ) -> None:
        """Define the marginals of interest with `marginal_indices` and the estimator architechture with `network`.

        Args:
            marginal_indices: marginals of interest defined by the parameter index
            network: a neural network which accepts `observation` and `parameters` and returns `len(marginal_indices)` ratios.
            device
        """
        self.marginal_indices = tupleize_marginal_indices(marginal_indices)
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
        max_epochs: int = 2**31 - 1,
        nworkers: int = 0,
        non_blocking: bool = True,
        pin_memory: bool = True,
    ) -> None:
        """Train the ratio estimator based off of a `dataset` containing observation and parameter pairs.

        Note: if the network has already been trained, training will resume where it left off.
        This effectively ignores `optimizer`, `learning_rate`, `scheduler`, and `scheduler_args`.

        Args:
            dataset: torch dataset which returns a tuple of (`observation`, `parameters`)
            batch_size
            learning_rate
            validation_percentage: Approximates the percentage of `dataset` used in the validation set
            optimizer: from `torch.optim` optimizer. It can only accept two arguments: `parameters` and `lr`. Need more arguments? Use `functools.partial`.
            scheduler: from `torch.optim.lr_scheduler`
            scheduler_kwargs: The arguments which get passed to `scheduler`
            early_stopping_patience: after this many fuitless epochs, training stops
            max_epochs: maximum number of epochs to train
            nworkers: number of workers to divide `dataloader` duties between. 0 implies one thread for training and dataloading.
            non_blocking: consult torch documentation, generally use `True`
            pin_memory: consult torch documentation, generally use `True`
        """
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
            for observation, _, v in train_loader:
                self.optimizer.zero_grad()
                observation = swyft.utils.dict_to_device(
                    observation, device=self.device, non_blocking=non_blocking
                )
                v = v.to(self.device)
                loss = self._loss(observation, v).sum(dim=0)
                loss.backward()
                self.optimizer.step()

            self.epoch += 1

            # Evaluation
            self.network.eval()
            loss_sum = 0
            with torch.no_grad():
                for observation, _, v in valid_loader:
                    observation = swyft.utils.dict_to_device(
                        observation, device=self.device, non_blocking=non_blocking
                    )
                    v = v.to(self.device)
                    validation_loss = self._loss(observation, v).sum(dim=0)
                    loss_sum += validation_loss
                loss_avg = loss_sum / n_validation_batches
                print(
                    "\rtraining: lr=%.2g, epoch=%i, validation loss=%.4g"
                    % (self._get_last_lr(self.scheduler), self.epoch, loss_avg),
                    end="",
                    flush=True,
                )
                validation_losses.append(loss_avg)

                if self.epoch == 0 or self.min_loss > loss_avg:
                    fruitless_epoch = 0
                    self.min_loss = loss_avg
                    self.best_network_state_dict = self.network.state_dict()
                else:
                    fruitless_epoch += 1

                if self.scheduler is not None:
                    self.scheduler.step(loss_avg)
        print("")
        return validation_losses

    def log_ratio(
        self,
        observation: ObsType,
        v: Array,
        batch_size: Optional[int] = None,
    ) -> MarginalToArray:
        """Evaluate the ratio estimator on a single `observation` with many `parameters`.
        The `parameters` correspond to `v`, i.e. the "physical" parameterization.
        (As opposed to `u` which is mapped to the hypercube.)

        Args:
            observation: a single observation to estimate ratios on (Cannot have a batch dimension!)
            v: parameters
            batch_size: divides the evaluation into batches of this size

        Returns:
            MarginalToArray: the ratios of each marginal in `marginal_indices`. Each marginal index is a key.
        """
        was_training = self.network.training
        self.network.eval()

        with torch.no_grad():
            observation = dict_array_to_tensor(observation, device=self.device)
            features = self.network.head(
                {key: value.unsqueeze(0) for key, value in observation.items()}
            )
            len_v = len(v)
            if batch_size is None or len_v <= batch_size:
                v = array_to_tensor(v, device=self.device)
                repeated_features = features.expand(v.size(0), *features.shape[1:])
                ratio = self.network.tail(repeated_features, v).cpu().numpy()
            else:
                ratio = []
                for i in range(len_v // batch_size + 1):
                    parameter_batch = array_to_tensor(
                        v[i * batch_size : (i + 1) * batch_size, :],
                        device=self.device,
                    )
                    feature_batch = features.expand(
                        parameter_batch.size(0), *features.shape[1:]
                    )
                    ratio_batch = (
                        self.network.tail(feature_batch, parameter_batch).cpu().numpy()
                    )
                    ratio.append(ratio_batch)
                ratio = np.vstack(ratio)

        if was_training:
            self.network.train()
        else:
            self.network.eval()

        return {k: ratio[..., i] for i, k in enumerate(self.marginal_indices)}

    @staticmethod
    def _repeat_observation_to_match_v(
        observation: Dict[Hashable, torch.Tensor], v: torch.Tensor
    ) -> Dict[Hashable, torch.Tensor]:
        b, *_ = v.size()
        return valmap(lambda x: x.unsqueeze(0).expand(b, *x.size()), observation)

    @staticmethod
    def _get_last_lr(scheduler: SchedulerType) -> float:
        """Get the last learning rate from a `lr_scheduler`."""
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
        """Calculate the marginal-wise losses.

        Args:
            observation: a batch of observations within a dictionary
            parameters: a batch of parameters

        Returns:
            torch.Tensor: the marginal-wise losses with `len(self.marginal_indices)`
        """
        n_batch = parameters.size(0)
        assert (
            n_batch % 2 == 0
        ), "Loss function can only handle even-numbered batch sizes."
        assert all(
            [value.size(0) == n_batch for value in observation.values()]
        ), "The observation batch_size must agree with the parameter batch_size."

        # Repeat interleave
        observation_doubled = valmap(double_observation, observation)
        parameters_doubled = double_parameters(parameters)

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
        """Instantiate a MarginalRatioEstimator from a state_dict, along with a few necessary python objects.

        Args:
            network: initialized network
            optimizer: same optimizer as used by saved model
            scheduler: same scheduler as used by saved model
            device
            state_dict

        Returns:
            MarginalRatioEstimatorType: loaded model
        """
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

        if optimizer is not None and state_dict["optimizer"] is None:
            raise FileNotFoundError(
                "There was no data about the optimizer in the state_dict"
            )
        elif optimizer is not None and state_dict["optimizer"] is not None:
            marginal_ratio_estimator.optimizer.load_state_dict(state_dict["optimizer"])
        else:
            pass

        if scheduler is not None and state_dict["scheduler"] is None:
            raise FileNotFoundError(
                "There was no data about the scheduler in the state_dict"
            )
        elif scheduler is not None and state_dict["scheduler"] is not None:
            marginal_ratio_estimator.scheduler.load_state_dict(state_dict["scheduler"])
        else:
            pass

        return marginal_ratio_estimator

    @classmethod
    def load(
        cls: Type[StateDictSaveableType],
        network: torch.nn.Module,
        device: Device,
        filename: PathType,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> StateDictSaveableType:
        sd = torch.load(filename)
        return cls.from_state_dict(
            network=network,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            state_dict=sd,
        )


if __name__ == "__main__":
    pass
