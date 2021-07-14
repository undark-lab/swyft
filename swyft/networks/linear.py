import math

import torch


# Inspired by: https://github.com/pytorch/pytorch/issues/36591
class LinearWithChannels(torch.nn.Module):
    def __init__(self, channels, in_features, out_features):
        super(LinearWithChannels, self).__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((channels, out_features, in_features))
        )
        self.bias = torch.nn.Parameter(torch.empty(channels, out_features))

        # Initialize weights
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        assert x.ndim >= 2, "Requires (..., channel, features) shape."
        x = x.unsqueeze(-1)
        result = torch.matmul(self.weight, x).squeeze(-1) + self.bias
        return result
