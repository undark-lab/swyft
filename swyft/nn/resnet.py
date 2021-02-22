from typing import Callable

import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

from .batchnorm import BatchNorm1dWithChannel
from .linear import LinearWithChannel


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
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [BatchNorm1dWithChannel(channels, features, eps=1e-3) for _ in range(2)]
            )
        self.linear_layers = nn.ModuleList(
            [LinearWithChannel(features, features, channels) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs):
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


class ResidualNet(nn.Module):
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
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.initial_layer = LinearWithChannel(in_features, hidden_features, channels)
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
        self.final_layer = LinearWithChannel(hidden_features, out_features, channels)

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final_layer(temps)
        return outputs
