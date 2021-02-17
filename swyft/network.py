# pylint: disable=no-member, not-callable, access-member-before-definition
import math
import numpy as np

import torch
import torch.nn as nn

from .utils import Module, verbosity


def _get_z_shape(param_list):
    return (len(param_list), max([len(c) for c in param_list]))


# TODO: Remove redundant combine functions
def _combine(params, param_list):
    """Combine parameters according to parameter list. Supports one batch dimension."""
    shape = params[list(params)[0]].shape
    device = params[list(params)[0]].device
    z_shape = _get_z_shape(param_list)
    if len(shape) == 0:  # No batching
        z = torch.zeros(z_shape).to(device)
        for i, c in enumerate(param_list):
            pars = torch.stack([params[k] for k in c]).T
            z[i, : pars.shape[0]] = pars
    else:  # Batching
        n = shape[0]
        z = torch.zeros((n,) + z_shape).to(device)
        for i, c in enumerate(param_list):
            pars = torch.stack([params[k] for k in c]).T
            z[:, i, : pars.shape[1]] = pars
    return z


class OnlineNormalizationLayer(nn.Module):
    def __init__(self, shape, stable: bool = False, epsilon: float = 1e-10):
        """Accumulate mean and variance online using the "parallel algorithm" algorithm from [1].

        Args:
            shape (tuple): shape of mean, variance, and std array. do not include batch dimension!
            stable (bool): (optional) compute using the stable version of the algorithm [1]
            epsilon (float): (optional) added to the computation of the standard deviation for numerical stability.

        References:
            [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        super().__init__()
        self.register_buffer("n", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_mean", torch.zeros(shape))
        self.register_buffer("_M2", torch.zeros(shape))
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.shape = shape
        self.stable = stable

    def _parallel_algorithm(self, x):
        assert x.shape[1:] == self.shape
        na = self.n.clone()
        nb = x.shape[0]
        nab = na + nb

        xa = self._mean.clone()
        xb = x.mean(dim=0)
        delta = xb - xa
        if self.stable:
            xab = (na * xa + nb * xb) / nab
        else:
            xab = xa + delta * nb / nab

        m2a = self._M2.clone()
        m2b = (
            x.var(dim=(0,), unbiased=False) * nb
        )  # do not use bessel's correction then multiply by total number of items in batch.
        m2ab = m2a + m2b + delta ** 2 * na * nb / nab
        return nab, xab, m2ab

    def forward(self, x):
        if self.training:
            self.n, self._mean, self._M2 = self._parallel_algorithm(x)
        return (x - self.mean) / self.std

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        if self.n > 1:
            return self._M2 / (self.n - 1)
        else:
            return torch.zeros_like(self._M2)

    @property
    def std(self):
        return torch.sqrt(self.var + self.epsilon)


# From: https://github.com/pytorch/pytorch/issues/36591
class LinearWithChannel(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()

        # initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, output_size, input_size))
        self.b = torch.nn.Parameter(torch.zeros(channel_size, output_size))

        # change weights to kaiming
        self.reset_parameters(self.w, self.b)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        x = x.unsqueeze(-1)
        result = torch.matmul(self.w, x).squeeze(-1) + self.b
        return result


class DefaultTail(Module):
    def __init__(
        self,
        n_features,
        param_list,
        n_tail_features=2,
        p=0.0,
        hidden_layers=[256, 256, 256],
        online_norm=True,
        param_transform=None,
        tail_features=False,
    ):
        super().__init__(
            n_features,
            param_list,
            n_tail_features=n_tail_features,
            p=p,
            hidden_layers=hidden_layers,
            online_norm=online_norm,
            param_transform=param_transform,
            tail_features=tail_features,
        )
        self.param_list = param_list

        n_channels, pdim = _get_z_shape(param_list)
        self.n_channels = n_channels
        self.tail_features = tail_features

        # Feature compressor
        if self.tail_features:
            n_hidden = 256
            self.fcA = LinearWithChannel(n_features, n_hidden, n_channels)
            self.fcB = LinearWithChannel(n_hidden, n_hidden, n_channels)
            self.fcC = LinearWithChannel(n_hidden, n_tail_features, n_channels)
        else:
            n_tail_features = n_features

        # Pre-network parameter transformation hook
        self.param_transform = param_transform

        # Online normalization of (transformed) parameters
        if online_norm:
            self.onl_z = OnlineNormalizationLayer(torch.Size([n_channels, pdim]))
        else:
            self.onl_z = lambda z: z

        # Ratio estimator
        if isinstance(p, float):
            p = [p for i in range(len(hidden_layers))]
        ratio_estimator_config = [
            LinearWithChannel(pdim + n_tail_features, hidden_layers[0], n_channels),
            nn.ReLU(),
            nn.Dropout(p=p[0]),
        ]
        for i in range(len(hidden_layers) - 1):
            ratio_estimator_config += [
                LinearWithChannel(hidden_layers[i], hidden_layers[i + 1], n_channels),
                nn.ReLU(),
                nn.Dropout(p=p[i + 1]),
            ]
        ratio_estimator_config += [LinearWithChannel(hidden_layers[-1], 1, n_channels)]
        self.ratio_estimator = nn.Sequential(*ratio_estimator_config)

        self.af = nn.ReLU()

    def forward(self, f, params):
        """Forward pass tail network.  Can handle one batch dimension.

        Args:
            f (tensor): feature vectors with shape (n_batch, n_features)
            params (dict): parameter dictionary, with parameter shape (n_batch,)

        Returns:
            lnL (tensor): lnL ratio with shape (n_batch, len(param_list))
        """
        # Parameter transform hook
        if self.param_transform is not None:
            params = self.param_transform(params)

        # Feature compressors independent per channel
        f = f.unsqueeze(1).repeat(
            1, self.n_channels, 1
        )  # (n_batch, n_channels, n_features)
        if self.tail_features:
            f = self.af(self.fcA(f))
            f = self.af(self.fcB(f))
            f = self.fcC(f)

        # Channeled density estimator
        z = _combine(params, self.param_list)
        z = self.onl_z(z)

        x = torch.cat([f, z], -1)
        x = self.ratio_estimator(x)
        x = x.squeeze(-1)
        return x


# FIXME: Remove obs_transform. This should not be required for anything.

class DefaultHead(Module):
    def __init__(self, obs_shapes, online_norm=True, obs_transform=None):
        super().__init__(
            obs_shapes=obs_shapes, obs_transform=obs_transform, online_norm=online_norm
        )
        self.obs_transform = obs_transform

        if not all(np.array([len(v) for v in obs_shapes.values()]) == 1):
            raise ValueError("DefaultHead only supports 1-dim data. Please supply custom head network.")

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


if __name__ == "__main__":
    pass
