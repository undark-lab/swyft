from typing import Dict, Hashable, Tuple

import torch
import torch.nn as nn

import swyft
import swyft.utils
from swyft.networks.channelized import ResidualNetWithChannel
from swyft.networks.standardization import OnlineStandardizingLayer
from swyft.types import Array, MarginalIndex


def is_marginal_block_possible(marginal_indices: MarginalIndex) -> bool:
    marginal_indices = swyft.utils.tupleize_marginals(marginal_indices)
    return [len(marginal_indices[0]) == len(mi) for mi in marginal_indices]


def get_marginal_block_shape(marginal_indices: MarginalIndex) -> Tuple[int, int]:
    marginal_indices = swyft.utils.tupleize_marginals(marginal_indices)
    assert is_marginal_block_possible(
        marginal_indices
    ), f"Each tuple in {marginal_indices} must have the same length."
    return len(marginal_indices), len(marginal_indices[0])


def get_marginal_block(
    parameters: Array, marginal_indices: MarginalIndex
) -> torch.Tensor:
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
    def __init__(self, observation_key: Hashable) -> None:
        super().__init__()
        self.observation_key = observation_key
        self.flatten = nn.Flatten()

    def forward(self, observation: Dict[Hashable, torch.Tensor]) -> torch.Tensor:
        return self.flatten(observation[self.observation_key])  # B, O


class ParameterTransform(nn.Module):
    def __init__(self, marginal_indices: MarginalIndex):
        super().__init__()
        self.register_buffer(
            "marginal_indices",
            torch.tensor(swyft.utils.tupleize_marginals(marginal_indices)),
        )

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        return get_marginal_block(parameters, self.marginal_indices)  # B, M, P


class MarginalClassifier(nn.Module):
    def __init__(
        self,
        n_marginals: int,
        n_combined_features: int,
        hidden_features: int,
        num_blocks: int,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.n_marginals = n_marginals
        self.n_combined_features = n_combined_features
        self.net = ResidualNetWithChannel(
            channels=self.n_marginals,
            in_features=self.n_combined_features,
            out_features=1,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    def forward(
        self, features: torch.Tensor, marginal_block: torch.Tensor
    ) -> torch.Tensor:
        fb = features.unsqueeze(1).expand(-1, self.n_marginals, -1)  # B, M, O
        combined = torch.cat([fb, marginal_block], dim=2)  # B, M, O + P
        return self.net(combined).squeeze(-1)  # B, M


class Network(nn.Module):
    def __init__(
        self,
        observation_transform: nn.Module,
        parameter_transform: nn.Module,
        marginal_classifier: nn.Module,
        n_parameters: int,
        observation_online_z_score: bool,
        parameter_online_z_score: bool,
    ) -> None:
        super().__init__()
        self.observation_transform = observation_transform
        self.parameter_transform = parameter_transform
        self.marginal_classifier = marginal_classifier
        self.n_parameters = n_parameters

        if observation_online_z_score:
            self.observation_online_z_score = OnlineStandardizingLayer(
                torch.Size([self.observation_transform.in_features])
            )
        else:
            self.observation_online_z_score = nn.Identity()

        if parameter_online_z_score:
            self.parameter_online_z_score = OnlineStandardizingLayer(
                torch.Size([self.n_parameters])
            )
        else:
            self.parameter_online_z_score = nn.Identity()

    def forward(
        self, observation: Dict[Hashable, torch.Tensor], parameters: torch.Tensor
    ) -> torch.Tensor:
        observation_z_scored = self.observation_online_z_score(observation)  # B, O
        parameters_z_scored = self.parameter_online_z_score(
            parameters
        )  # B, n_parameters

        features = self.observation_transform(observation_z_scored)  # B, O
        marginal_block = self.parameter_transform(parameters_z_scored)  # B, M, P
        return self.marginal_classifier(features, marginal_block)  # B, M


def get_marginal_classifier(
    observation_key: Hashable,
    marginal_indices: MarginalIndex,
    n_observation_features: int,
    n_parameters: int,
    hidden_features: int,
) -> nn.Module:
    n_marginals, n_block_parameters = get_marginal_block_shape(marginal_indices)
    return Network(
        ObservationTransform(observation_key),
        ParameterTransform(marginal_indices),
        MarginalClassifier(
            n_marginals,
            n_observation_features + n_block_parameters,
            hidden_features=hidden_features,
        ),
        n_parameters,
        observation_online_z_score=True,
        parameter_online_z_score=True,
    )


if __name__ == "__main__":
    pass
