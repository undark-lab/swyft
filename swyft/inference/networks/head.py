import numpy as np
import torch

from swyft.nn.module import Module
from swyft.nn.normalization import OnlineNormalizationLayer


# FIXME: Remove obs_transform. This should not be required for anything.
class DefaultHead(Module):
    def __init__(self, obs_shapes, online_norm=True, obs_transform=None):
        super().__init__(
            obs_shapes=obs_shapes, obs_transform=obs_transform, online_norm=online_norm
        )
        self.obs_transform = obs_transform

        if not all(np.array([len(v) for v in obs_shapes.values()]) == 1):
            raise ValueError(
                "DefaultHead only supports 1-dim data. Please supply custom head network."
            )

        self.n_features = sum([v[0] for k, v in obs_shapes.items()])

        if online_norm:
            self.onl_f = OnlineNormalizationLayer(torch.Size([self.n_features]))
        else:
            self.onl_f = lambda f: f

    def forward(self, obs):
        """Forward pass default head network. Concatenate.

        Args:
            obs (dict): Dictionary of tensors with shape (n_batch, m_i)

        Returns:
            f (tensor): Feature vectors with shape (n_batch, M), with M = sum_i m_i
        """
        if self.obs_transform is not None:
            obs = self.obs_transform(obs)
        f = []
        for key, value in sorted(obs.items()):
            f.append(value)
        f = torch.cat(f, dim=-1)
        f = self.onl_f(f)
        return f
