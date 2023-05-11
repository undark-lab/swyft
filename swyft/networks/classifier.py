from abc import ABC, abstractmethod
from typing import Dict, Hashable, Tuple

import torch
import torch.nn as nn

import swyft
import swyft.utils
from swyft.networks.channelized import ResidualNetWithChannel
from swyft.networks.standardization import (
    OnlineDictStandardizingLayer,
    OnlineStandardizingLayer,
)
from swyft.types import Array, MarginalIndex, ObsShapeType


class HeadTailClassifier(ABC):
    """Abstract class which ensures that child classifier networks will function with swyft"""

    @abstractmethod
    def head(self, observation: Dict[Hashable, torch.Tensor]) -> torch.Tensor:
        """convert the observation into a tensor of features

        Args:
            observation: observation type

        Returns:
            a tensor of features which can be utilized by tail
        """
        pass

    @abstractmethod
    def tail(self, features: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """finish the forward pass using features computed by head

        Args:
            features: output of head
            parameters: the parameters normally given to forward pass

        Returns:
            the same output as `forward(observation, parameters)`
        """
        pass


class ObservationTransform(nn.Module):
    def __init__(
        self,
        observation_key: Hashable,
        observation_shapes: ObsShapeType,
        online_z_score: bool,
    ) -> None:
        super().__init__()
        self.observation_key = observation_key
        self.observation_shapes = observation_shapes
        self.flatten = nn.Flatten()
        if online_z_score:
            self.online_z_score = OnlineDictStandardizingLayer(self.observation_shapes)
        else:
            self.online_z_score = nn.Identity()

    def forward(self, observation: Dict[Hashable, torch.Tensor]) -> torch.Tensor:
        z_scored_observation = self.online_z_score(observation)
        return self.flatten(z_scored_observation[self.observation_key])  # B, O

    @property
    def n_features(self) -> int:
        with torch.no_grad():
            fabricated_observation = {
                key: torch.rand(2, *shape)
                for key, shape in self.observation_shapes.items()
            }
            _, n_features = self.forward(fabricated_observation).shape
        return n_features


class ParameterTransform(nn.Module):
    def __init__(
        self, n_parameters: int, marginal_indices: MarginalIndex, online_z_score: bool
    ) -> None:
        super().__init__()
        self.register_buffer(
            "marginal_indices",
            torch.tensor(swyft.utils.tupleize_marginal_indices(marginal_indices)),
        )
        self.n_parameters = torch.Size([n_parameters])
        if online_z_score:
            self.online_z_score = OnlineStandardizingLayer(self.n_parameters)
        else:
            self.online_z_score = nn.Identity()

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        parameters = self.online_z_score(parameters)
        return self.get_marginal_block(parameters, self.marginal_indices)  # B, M, P

    @property
    def marginal_block_shape(self) -> Tuple[int, int]:
        return self.get_marginal_block_shape(self.marginal_indices)

    @staticmethod
    def is_marginal_block_possible(marginal_indices: MarginalIndex) -> bool:
        marginal_indices = swyft.utils.tupleize_marginal_indices(marginal_indices)
        return [len(marginal_indices[0]) == len(mi) for mi in marginal_indices]

    @classmethod
    def get_marginal_block_shape(
        cls, marginal_indices: MarginalIndex
    ) -> Tuple[int, int]:
        marginal_indices = swyft.utils.tupleize_marginal_indices(marginal_indices)
        assert cls.is_marginal_block_possible(
            marginal_indices
        ), f"Each tuple in {marginal_indices} must have the same length."
        return len(marginal_indices), len(marginal_indices[0])

    @classmethod
    def get_marginal_block(
        cls, parameters: Array, marginal_indices: MarginalIndex
    ) -> torch.Tensor:
        depth = swyft.utils.depth(marginal_indices)
        tuple_marginal_indices = swyft.utils.tupleize_marginal_indices(marginal_indices)
        assert cls.is_marginal_block_possible(
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


def spectral_embedding(z, Lmax=8):
    device = z.device
    DB = z.shape[-1]
    f = 2 ** torch.arange(Lmax, device=device)
    ZF = z.repeat_interleave(Lmax, dim=-1) * f.repeat(DB)
    # Embedding multiplies last dimension size by 2*Lmax+1
    return torch.cat([z, torch.sin(ZF), torch.cos(ZF)], dim=-1)


class MarginalClassifier(nn.Module):
    def __init__(
        self,
        n_marginals: int,
        n_combined_features: int,
        hidden_features: int,
        num_blocks: int,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = True,
        Lmax: int = 0,
    ) -> None:
        super().__init__()
        self.n_marginals = n_marginals
        self.n_combined_features = n_combined_features
        self.net = ResidualNetWithChannel(
            channels=self.n_marginals,
            in_features=self.n_combined_features * (1 + 2 * Lmax),
            out_features=1,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self.Lmax = Lmax

    def forward(
        self, features: torch.Tensor, marginal_block: torch.Tensor
    ) -> torch.Tensor:
        if len(features.shape) == 2:  # Input shape is B, O
            fb = features.unsqueeze(1).expand(-1, self.n_marginals, -1)  # B, M, O
        else:
            fb = features  # Input shape is alreadby B, M, O
        combined = torch.cat([fb, marginal_block], dim=2)  # B, M, O + P
        if self.Lmax > 0:
            combined = spectral_embedding(combined, Lmax=self.Lmax)
        return self.net(combined).squeeze(-1)  # B, M


class Network(nn.Module, HeadTailClassifier):
    def __init__(
        self,
        observation_transform: nn.Module,
        parameter_transform: nn.Module,
        marginal_classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.observation_transform = observation_transform
        self.parameter_transform = parameter_transform
        self.marginal_classifier = marginal_classifier

    def forward(
        self, observation: Dict[Hashable, torch.Tensor], parameters: torch.Tensor
    ) -> torch.Tensor:
        features = self.observation_transform(observation)  # B, O
        marginal_block = self.parameter_transform(parameters)  # B, M, P
        return self.marginal_classifier(features, marginal_block)  # B, M

    def head(self, observation: Dict[Hashable, torch.Tensor]) -> torch.Tensor:
        return self.observation_transform(observation)  # B, O

    def tail(self, features: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        marginal_block = self.parameter_transform(parameters)  # B, M, P
        return self.marginal_classifier(features, marginal_block)  # B, M


def get_marginal_classifier(
    observation_key: Hashable,
    marginal_indices: MarginalIndex,
    observation_shapes: ObsShapeType,
    n_parameters: int,
    hidden_features: int,
    num_blocks: int,
    observation_online_z_score: bool = True,
    parameter_online_z_score: bool = True,
) -> nn.Module:
    observation_transform = ObservationTransform(
        observation_key, observation_shapes, online_z_score=observation_online_z_score
    )
    n_observation_features = observation_transform.n_features

    parameter_transform = ParameterTransform(
        n_parameters, marginal_indices, online_z_score=parameter_online_z_score
    )
    n_marginals, n_block_parameters = parameter_transform.marginal_block_shape

    marginal_classifier = MarginalClassifier(
        n_marginals,
        n_observation_features + n_block_parameters,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
    )

    return Network(
        observation_transform,
        parameter_transform,
        marginal_classifier,
    )


if __name__ == "__main__":
    pass
