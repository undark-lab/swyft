import math
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


# Inspired by: https://github.com/pytorch/pytorch/issues/36591
class LinearWithChannel(torch.nn.Module):
    def __init__(self, channels: int, in_features: int, out_features: int) -> None:
        super(LinearWithChannel, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.empty((channels, out_features, in_features))
        )
        self.bias = torch.nn.Parameter(torch.empty(channels, out_features))

        # Initialize weights
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 2, "Requires (..., channel, features) shape."
        x = x.unsqueeze(-1)
        result = torch.matmul(self.weights, x).squeeze(-1) + self.bias
        return result


class BatchNorm1dWithChannel(nn.BatchNorm1d):
    def __init__(
        self,
        num_channels: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        """BatchNorm1d over the batch, N. Requires shape (N, C, L).

        Otherwise, same as torch.nn.BatchNorm1d with extra num_channel. Cannot do the temporal batch norm case.
        """
        num_features = num_channels * num_features
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.flatten = nn.Flatten()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        n, c, f = input.shape
        flat = self.flatten(input)
        batch_normed = super().forward(flat)
        return batch_normed.reshape(n, c, f)


# inspired by https://github.com/bayesiains/nflows/blob/master/nflows/nn/nets/resnet.py
class ResidualBlockWithChannel(nn.Module):
    """A general-purpose residual block. Works only with channelized 1-dim inputs."""

    def __init__(
        self,
        channels: int,
        features: int,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        zero_initialization: bool = True,
    ) -> None:
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [BatchNorm1dWithChannel(channels, features, eps=1e-3) for _ in range(2)]
            )
        self.linear_layers = nn.ModuleList(
            [LinearWithChannel(channels, features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weights, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        return inputs + temps


# inspired by https://github.com/bayesiains/nflows/blob/master/nflows/nn/nets/resnet.py
class ResidualNetWithChannel(nn.Module):
    """A general-purpose residual network. Works only with channelized 1-dim inputs."""

    def __init__(
        self,
        channels: int,
        in_features: int,
        out_features: int,
        hidden_features: int,
        num_blocks: int = 2,
        activation: Callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.initial_layer = LinearWithChannel(channels, in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlockWithChannel(
                    channels=channels,
                    features=hidden_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = LinearWithChannel(channels, hidden_features, out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final_layer(temps)
        return outputs
