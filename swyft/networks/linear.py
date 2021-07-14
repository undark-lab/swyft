import math

import torch


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
