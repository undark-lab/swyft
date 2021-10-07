from typing import Dict, Hashable, Tuple

import torch
import torch.nn as nn

import swyft
import swyft.utils
from swyft.networks.channelized import LinearWithChannel
from swyft.networks.normalization import OnlineNormalizationLayer
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
        observation_online_z_score: bool = True,
        parameter_online_z_score: bool = True,
    ) -> None:
        super().__init__()
        self.observation_transform = observation_transform
        self.parameter_transform = parameter_transform
        self.classifier = classifier

        if observation_online_z_score:
            self.observation_online_z_score = OnlineNormalizationLayer(
                torch.Size([self.observation_transform.in_features])
            )
        else:
            self.observation_online_z_score = nn.Identity()

        if parameter_online_z_score:
            raise NotImplementedError("TODO")
            get_marginal_block_shape(self.parameter_transform.marginal_indices)
            self.parameter_online_z_score = OnlineNormalizationLayer()
            # TODO need to access n_parameters.
            # alternatively, create a z_scoring network which already takes shaped data.
        else:
            self.parameter_online_z_score = nn.Identity()

    def forward(
        self, observation: Dict[Hashable, torch.Tensor], parameters: torch.Tensor
    ) -> torch.Tensor:
        observation_z_scored = self.observation_online_z_score(observation)  # B, O
        parameters_z_scored = self.parameter_online_z_score(
            parameters
        )  # B, n_parameters

        features = self.observation_transform(observation_z_scored)  # B, F
        marginal_block = self.parameter_transform(parameters_z_scored)  # B, M, P
        return self.classifier(features, marginal_block)  # B, M


def get_classifier(
    observation_key: Hashable,
    marginal_indices: MarginalIndex,
    n_observation_features: int,
    n_observation_embedding_features: int,
) -> nn.Module:
    nm, nbp = get_marginal_block_shape(marginal_indices)
    n_observation_embedding_features = 50  # TODO flexible
    return Network(
        ObservationTransform(
            observation_key, n_observation_features, n_observation_embedding_features
        ),
        ParameterTransform(marginal_indices),
        Classifier(nm, n_observation_embedding_features + nbp),
    )


if __name__ == "__main__":
    pass
