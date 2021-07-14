import numpy as np
import torch

from swyft.networks.module import Module
from swyft.networks.normalization import OnlineNormalizationLayer
from swyft.types import ObsType, SimShapeType


class DefaultHead(Module):
    """Default head network.

    Args:
        sim_shapes: Shape of the simulation data
        online_norm: Perform online normalization of the inputs

    .. note::
        The default head network requires that all simulation components are
        1-dim.  They will be simply concatenated into a single (potentially
        very large) feature vector.  `DefaultHead` should only be used for very
        low-dimensional data.  Almost always custom implementations will lead
        to better results.
    """

    def __init__(self, sim_shapes: SimShapeType, online_norm: bool = True) -> None:
        super().__init__(sim_shapes=sim_shapes, online_norm=online_norm)
        if not all(np.array([len(v) for v in sim_shapes.values()]) == 1):
            raise ValueError(
                "DefaultHead only supports 1-dim data. Please supply custom head network."
            )

        self.n_features = sum([v[0] for k, v in sim_shapes.items()])

        if online_norm:
            self.onl_f = OnlineNormalizationLayer(torch.Size([self.n_features]))
        else:
            self.onl_f = lambda f: f

    def forward(self, sim: ObsType) -> torch.Tensor:
        """Forward pass default head network. Concatenate.

        Args:
            sim: Dictionary of tensors with shape (n_batch, m_i)

        Returns:
            f: Feature vectors with shape (n_batch, M), with M = sum_i m_i
        """
        f = []
        for key, value in sorted(sim.items()):
            f.append(value)
        f = torch.cat(f, dim=-1)
        f = self.onl_f(f)
        return f
