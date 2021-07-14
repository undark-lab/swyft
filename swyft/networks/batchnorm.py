import torch
import torch.nn as nn


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
