from typing import Callable, Dict, Hashable, Optional, Tuple

import torch
import torch.nn as nn
from toolz.dicttoolz import valmap
from torch.utils.data import DataLoader, Dataset, random_split

import swyft
import swyft.utils
from swyft.inference.loss import double_features, double_params
from swyft.inference.train import get_ntrain_nvalid
from swyft.networks.channelized import LinearWithChannel
from swyft.types import Array, Device, MarginalIndex


def is_marginal_block_possible(marginal_indices: MarginalIndex) -> bool:
    marginal_indices = swyft.utils.tupleize_marginals(marginal_indices)
    return [len(marginal_indices[0]) == len(mi) for mi in marginal_indices]


def get_marginal_block_shape(marginal_indices: MarginalIndex) -> Tuple[int, int]:
    marginal_indices = swyft.utils.tupleize_marginals(marginal_indices)
    assert is_marginal_block_possible(
        marginal_indices
    ), f"Each tuple in {marginal_indices} must have the same length."
    return len(marginal_indices), len(marginal_indices[0])


def get_marginal_block(parameters: Array, marginal_indices: MarginalIndex) -> Array:
    depth = swyft.utils.depth(marginal_indices)
    tuple_marginal_indices = swyft.utils.tupleize_marginals(marginal_indices)
    assert is_marginal_block_possible(
        tuple_marginal_indices
    ), f"Each tuple in {tuple_marginal_indices} must have the same length."

    if depth in [0, 1, 2]:
        return torch.stack(
            [parameters[..., mi] for mi in tuple_marginal_indices], dim=1
        )
    else:
        raise ValueError(
            f"{marginal_indices} must be of the form (a) 2, (b) [2, 3], (c) [2, [1, 3]], or (d) [[0, 1], [1, 2]]."
        )


class ObservationTransform(nn.Module):
    def __init__(
        self, observation_key: Hashable, in_features: int, out_features: int
    ) -> None:
        super().__init__()
        self.observation_key = observation_key
        self.in_features = in_features
        self.out_features = out_features
        self.net = nn.Linear(in_features, out_features)

    def forward(self, observation: Dict[Hashable, torch.Tensor]) -> torch.Tensor:
        x = observation[self.observation_key].flatten(1)  # B, O
        return self.net(x)  # B, F


class ParameterTransform(nn.Module):
    def __init__(self, marginal_indices: MarginalIndex):
        super().__init__()
        self.register_buffer(
            "marginal_indices",
            torch.tensor(swyft.utils.tupleize_marginals(marginal_indices)),
        )

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        return get_marginal_block(parameters, self.marginal_indices)  # B, M, P


class Classifier(nn.Module):
    def __init__(self, n_marginals: int, n_combined_features: int) -> None:
        super().__init__()
        self.n_marginals = n_marginals
        self.n_combined_features = n_combined_features
        self.net = LinearWithChannel(self.n_marginals, self.n_combined_features, 1)

    def forward(
        self, features: torch.Tensor, marginal_block: torch.Tensor
    ) -> torch.Tensor:
        fb = features.unsqueeze(1).expand(-1, self.n_marginals, -1)  # B, M, F
        combined = torch.cat([fb, marginal_block], dim=2)  # B, M, F + P
        return self.net(combined).squeeze(-1)  # B, M


class Network(nn.Module):
    def __init__(
        self,
        observation_transform: nn.Module,
        parameter_transform: nn.Module,
        classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.observation_transform = observation_transform
        self.parameter_transform = parameter_transform
        self.classifier = classifier

    def forward(
        self, observation: Dict[Hashable, torch.Tensor], parameters: torch.Tensor
    ) -> torch.Tensor:
        features = self.observation_transform(observation)  # B, F
        marginal_block = self.parameter_transform(parameters)  # B, M, P
        return self.classifier(features, marginal_block)  # B, M


def get_classifier(
    observation_key: Hashable,
    marginal_indices: MarginalIndex,
    n_observation_features: int,
    n_observation_embedding_features: int,
) -> nn.Module:
    nm, nbp = get_marginal_block_shape(marginal_indices)
    n_observation_embedding_features = 50
    return Network(
        ObservationTransform(
            observation_key, n_observation_features, n_observation_embedding_features
        ),
        ParameterTransform(marginal_indices),
        Classifier(nm, n_observation_embedding_features + nbp),
    )


################################
# MRE
################################


class MarginalRatioEstimator:
    def __init__(
        self,
        dataset: Dataset,
        marginal_indices: MarginalIndex,
        network: nn.Module,
        device: Device,
    ) -> None:
        self.dataset = dataset
        self.marginal_indices = marginal_indices
        self.network = network
        self.device = device

        self.optimizer = None
        self.scheduler = None
        self.epoch = None
        self.best_state_dict = None
        self.min_loss = float("-Inf")

    def train(
        self,
        batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_percentage: float = 0.1,
        optimizer: Callable = torch.optim.Adam,
        scheduler: Optional[Callable] = torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs: dict = {"factor": 0.1, "patience": 5},
        early_stopping_patience: Optional[int] = None,
        max_epochs: int = 2**31 - 1,
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

        n_train, n_valid = get_ntrain_nvalid(validation_percentage, len(self.dataset))
        dataset_train, dataset_valid = random_split(self.dataset, [n_train, n_valid])
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
                    % (learning_rate, self.epoch, loss_avg),
                    end="",
                    flush=True,
                )
                validation_losses.append(loss_avg)

                if self.epoch == 0 or self.min_loss > validation_loss:
                    fruitless_epoch = 0
                    self.min_loss = validation_loss
                    self.best_state_dict = self.network.state_dict()
                else:
                    fruitless_epoch += 1

                if self.scheduler is not None:
                    self.scheduler.step(loss_avg)

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


####################
# Simple Data
####################


class Data(torch.utils.data.Dataset):
    def __init__(self, x, theta) -> None:
        super().__init__()
        self.x = x
        self.theta = theta

    def __getitem__(self, index):
        return (
            {"x": self.x["x"][index]},  # x
            self.theta[index],  # u
            self.theta[index],  # v
        )

    def __len__(self):
        return len(self.theta)


def main():
    b, o = 10, 7
    f = 10
    p = 5
    marginal_indices = [1, 2, 3]
    key = "x"
    x = {key: torch.rand(b, o)}
    theta = torch.rand(b, p)

    device = "cpu"

    N = 10_000
    dataset = Data({"x": torch.rand(N, o)}, torch.rand(N, p))

    network = get_classifier(key, marginal_indices, o, f)
    val = network(x, theta)

    mre = MarginalRatioEstimator(
        dataset,
        marginal_indices,
        network,
        device,
    )
    mre.train()


if __name__ == "__main__":
    main()
