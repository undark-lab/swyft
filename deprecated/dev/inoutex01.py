from typing import Dict, Hashable, Tuple

import torch
import torch.nn as nn

import swyft
import swyft.utils
from swyft.networks.linear import LinearWithChannel
from swyft.types import Array, MarginalIndex


def is_marginal_block_possible(marginal_indices: MarginalIndex) -> bool:
    marginal_indices = swyft.utils.tupleize_marginals(marginal_indices)
    return [len(marginal_indices[0]) == len(mi) for mi in marginal_indices]


def get_marginal_block_shape(marginal_indices: MarginalIndex) -> Tuple[int, int]:
    marginal_indices = swyft.utils.tupleize_marginals(marginal_indices)
    assert is_marginal_block_possible(
        marginal_indices
    ), f"Each tuple in {marginal_indices=} must have the same length."
    return len(marginal_indices), len(marginal_indices[0])


def get_marginal_block(parameters: Array, marginal_indices: MarginalIndex) -> Array:
    depth = swyft.utils.depth(marginal_indices)
    tuple_marginal_indices = swyft.utils.tupleize_marginals(marginal_indices)
    assert is_marginal_block_possible(
        tuple_marginal_indices
    ), f"Each tuple in {tuple_marginal_indices=} must have the same length."

    if depth in [0, 1, 2]:
        return torch.stack(
            [parameters[..., mi] for mi in tuple_marginal_indices], dim=1
        )
    else:
        raise ValueError(
            f"{marginal_indices=} must be of the form (a) 2, (b) [2, 3], (c) [2, [1, 3]], or (d) [[0, 1], [1, 2]]."
        )


class Head(nn.Module):
    def __init__(
        self, observation_key: Hashable, in_features: int, out_features: int
    ) -> None:
        super().__init__()
        self.observation_key = observation_key
        self.in_features = in_features
        self.out_features = out_features
        self.net = nn.Linear(in_features, out_features)

    def forward(self, observation: Dict[Hashable, torch.Tensor]) -> torch.Tensor:
        x = observation[self.observation_key]
        return self.net(x)


class Combinator(nn.Module):
    def __init__(self, head_features: int, marginal_indices: MarginalIndex) -> None:
        super().__init__()
        self.head_features = head_features
        self.marginal_indices = marginal_indices
        self.n_marginals, self.n_parameters = get_marginal_block_shape(
            self.marginal_indices
        )

    def forward(self, features: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        mb = get_marginal_block(parameters, self.marginal_indices)  # B, M, P
        fb = features.unsqueeze(1).expand(-1, self.n_marginals, -1)  # B, M, F
        return torch.cat([fb, mb], dim=2)  # B, M, F + P


class Tail(nn.Module):
    def __init__(self, n_marginals: int, n_combined_features: int) -> None:
        super().__init__()
        self.n_marginals = n_marginals
        self.n_combined_features = n_combined_features
        self.net = LinearWithChannel(self.n_marginals, self.n_combined_features, 1)

    def forward(self, combined_features) -> torch.Tensor:
        print(combined_features.shape)
        return self.net(combined_features).squeeze(-1)


class MarginalLogRatioEstimator(nn.Module):
    def __init__(self, head: nn.Module, combinator: nn.Module, tail: nn.Module) -> None:
        super().__init__()
        self.head = head
        self.combinator = combinator
        self.tail = tail

    def forward(
        self, observation: Dict[Hashable, torch.Tensor], parameters: torch.Tensor
    ) -> torch.Tensor:
        h_out = self.head(observation)
        c_out = self.combinator(h_out, parameters)
        return self.tail(c_out)


def main():
    b, o = 10, 7
    f = 10
    p = 5
    marginal_indices = [1, 2, 3]
    key = "x"
    x = {key: torch.rand(b, o)}
    theta = torch.rand(b, p)

    nm, nbp = get_marginal_block_shape(marginal_indices)

    mre = MarginalLogRatioEstimator(
        Head(key, o, f),
        Combinator(f, marginal_indices),
        Tail(nm, f + nbp),
    )
    out = mre(x, theta)
    breakpoint()


if __name__ == "__main__":
    main()
