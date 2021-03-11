import math

import torch
import torch.nn as nn


# From: https://github.com/pytorch/pytorch/issues/36591
class LinearWithChannel(nn.Module):
    def __init__(self, channel_size, input_size, output_size):
        super(LinearWithChannel, self).__init__()

        # initialize weights
        self.weight = torch.nn.Parameter(
            torch.zeros(channel_size, output_size, input_size)
        )
        self.bias = torch.nn.Parameter(torch.zeros(channel_size, output_size))

        # change weights to kaiming
        self.reset_parameters(self.weight, self.bias)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        assert x.ndim >= 2, "Requires (..., channel, features) shape."
        x = x.unsqueeze(-1)
        result = torch.matmul(self.weight, x).squeeze(-1) + self.bias
        return result
