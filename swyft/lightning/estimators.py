from typing import (
    Callable,
    Dict,
    Hashable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import numpy as np
import torch
import torch.nn as nn
import swyft.networks

from swyft.lightning.core import *


def equalize_tensors(a, b):
    """Equalize tensors, for matching minibatch size of A and B."""
    n, m = len(a), len(b)
    if n == m:
        return a, b
    elif n == 1:
        shape = list(a.shape)
        shape[0] = m
        return a.expand(*shape), b
    elif m == 1:
        shape = list(b.shape)
        shape[0] = n
        return a, b.expand(*shape)
    elif n < m:
        assert m % n == 0, "Cannot equalize tensors with non-divisible batch sizes."
        shape = [1 for _ in range(a.dim())]
        shape[0] = m // n
        return a.repeat(*shape), b
    else:
        assert n % m == 0, "Cannot equalize tensors with non-divisible batch sizes."
        shape = [1 for _ in range(b.dim())]
        shape[0] = n // m
        return a, b.repeat(*shape)


class LogRatioEstimator_Ndim(torch.nn.Module):
    """Channeled MLPs for estimating multi-dimensional posteriors."""

    def __init__(
        self,
        num_features,
        marginals,
        varnames=None,
        dropout=0.1,
        hidden_features=64,
        num_blocks=2,
    ):
        super().__init__()
        self.marginals = marginals
        self.ptrans = swyft.networks.ParameterTransform(
            len(marginals), marginals, online_z_score=False
        )
        n_marginals, n_block_parameters = self.ptrans.marginal_block_shape
        n_observation_features = num_features
        self.classifier = swyft.networks.MarginalClassifier(
            n_marginals,
            n_observation_features + n_block_parameters,
            hidden_features=hidden_features,
            dropout_probability=dropout,
            num_blocks=num_blocks,
        )
        if isinstance(varnames, str):
            basename = varnames
            varnames = []
            for marg in marginals:
                varnames.append([basename + "[%i]" % i for i in marg])
        self.varnames = varnames

    def forward(self, x, z):
        x, z = equalize_tensors(x, z)
        z = self.ptrans(z)
        ratios = self.classifier(x, z)
        w = LogRatioSamples(
            ratios,
            z,
            np.array(self.varnames),
            metadata={"type": "MarginalMLP", "marginals": self.marginals},
        )
        return w


# TODO: Introduce RatioEstimatorDense
class _RatioEstimatorMLPnd(torch.nn.Module):
    def __init__(self, x_dim, marginals, dropout=0.1, hidden_features=64, num_blocks=2):
        super().__init__()
        self.marginals = marginals
        self.ptrans = swyft.networks.ParameterTransform(
            len(marginals), marginals, online_z_score=False
        )
        n_marginals, n_block_parameters = self.ptrans.marginal_block_shape
        n_observation_features = x_dim
        self.classifier = swyft.networks.MarginalClassifier(
            n_marginals,
            n_observation_features + n_block_parameters,
            hidden_features=hidden_features,
            dropout_probability=dropout,
            num_blocks=num_blocks,
        )

    def forward(self, x, z):
        x, z = equalize_tensors(x, z)
        z = self.ptrans(z)
        ratios = self.classifier(x, z)
        w = LogRatioSamples(
            ratios, z, metadata={"type": "MarginalMLP", "marginals": self.marginals}
        )
        return w


# TODO: Deprecated class (reason: Change of name)
class _RatioEstimatorMLP1d(torch.nn.Module):
    def __init__(
        self,
        x_dim,
        z_dim,
        varname=None,
        varnames=None,
        dropout=0.1,
        hidden_features=64,
        num_blocks=2,
        use_batch_norm=True,
        ptrans_online_z_score=True,
    ):
        """
        Default module for estimating 1-dim marginal posteriors.

        Args:
            x_dim: Length of feature vector.
            z_dim: Length of parameter vector.
            varnames: List of name of parameter vector. If a single string is provided, indices are attached automatically.
        """
        print("WARNING: Deprecated, use LogRatioEstimator_1dim instead.")
        super().__init__()
        self.marginals = [(i,) for i in range(z_dim)]
        self.ptrans = swyft.networks.ParameterTransform(
            len(self.marginals), self.marginals, online_z_score=ptrans_online_z_score
        )
        n_marginals, n_block_parameters = self.ptrans.marginal_block_shape
        n_observation_features = x_dim
        self.classifier = swyft.networks.MarginalClassifier(
            n_marginals,
            n_observation_features + n_block_parameters,
            hidden_features=hidden_features,
            dropout_probability=dropout,
            num_blocks=num_blocks,
            use_batch_norm=use_batch_norm,
        )
        if isinstance(varnames, list):
            self.varnames = np.array(varnames)
        else:
            self.varnames = np.array([varnames + "[%i]" % i for i in range(z_dim)])

    def forward(self, x, z):
        x, z = equalize_tensors(x, z)
        zt = self.ptrans(z).detach()
        logratios = self.classifier(x, zt)
        w = LogRatioSamples(logratios, z, self.varnames, metadata={"type": "MLP1d"})
        return w


class LogRatioEstimator_1dim(torch.nn.Module):
    """Channeled MLPs for estimating one-dimensional posteriors.

    Args:
        num_features: Number of features
    """

    def __init__(
        self,
        num_features: int,
        num_params,
        varnames=None,
        dropout=0.1,
        hidden_features=64,
        num_blocks=2,
        use_batch_norm=True,
        ptrans_online_z_score=True,
    ):
        """
        Default module for estimating 1-dim marginal posteriors.

        Args:
            num_features: Length of feature vector.
            num_params: Length of parameter vector.
            varnames: List of name of parameter vector. If a single string is provided, indices are attached automatically.
        """
        super().__init__()
        self.marginals = [(i,) for i in range(num_params)]
        self.ptrans = swyft.networks.ParameterTransform(
            len(self.marginals), self.marginals, online_z_score=ptrans_online_z_score
        )
        n_marginals, n_block_parameters = self.ptrans.marginal_block_shape
        n_observation_features = num_features
        self.classifier = swyft.networks.MarginalClassifier(
            n_marginals,
            n_observation_features + n_block_parameters,
            hidden_features=hidden_features,
            dropout_probability=dropout,
            num_blocks=num_blocks,
            use_batch_norm=use_batch_norm,
        )
        if isinstance(varnames, list):
            self.varnames = np.array([[v] for v in varnames])
        else:
            self.varnames = np.array(
                [[varnames + "[%i]" % i] for i in range(num_params)]
            )

    def forward(self, x, z):
        x, z = equalize_tensors(x, z)
        zt = self.ptrans(z).detach()
        logratios = self.classifier(x, zt)
        w = LogRatioSamples(
            logratios, z.unsqueeze(-1), self.varnames, metadata={"type": "MLP1d"}
        )
        return w


class LogRatioEstimator_1dim_Gaussian(torch.nn.Module):
    """Estimating posteriors assuming that they are Gaussian."""

    def __init__(
        self, num_params, varnames=None, momentum: float = 0.1, minstd: float = 1e-3
    ):
        r"""
        Default module for estimating 1-dim marginal posteriors, using Gaussian approximations.

        Args:
            num_params: Length of parameter vector.
            varnames: List of name of parameter vector. If a single string is provided, indices are attached automatically.
            momentum: Momentum for running estimate for variance and covariances.
            minstd: Minimum relative standard deviation of prediction variable.
                The correlation coefficient will be truncated in the range :math:`\rho = \pm \sqrt{1-\text{minstd}^2}`

        .. note::

           This module performs running estimates of parameter variances and
           covariances.  There are no learnable parameters.  This can cause errors when using the module
           in isolation without other modules with learnable parameters.

           The covariance estimates are based on joined samples only.  The
           first n_batch samples of z are assumed to be joined jointly drawn, where n_batch is the batch
           size of x.
        """
        super().__init__()
        self.momentum = momentum
        self.x_mean = None
        self.z_mean = None
        self.x_var = None
        self.z_var = None
        self.xz_cov = None
        self.minstd = minstd

        if isinstance(varnames, list):
            self.varnames = np.array([[v] for v in varnames])
        else:
            self.varnames = np.array(
                [[varnames + "[%i]" % i] for i in range(num_params)]
            )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """2-dim Gaussian approximation to marginals and joint, assuming (B, N)."""
        if self.training or self.x_mean is None:
            batch_size = len(x)
            idx = np.arange(batch_size)

            # Estimation w/o Bessel's correction, using simple MLE estimate (https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices)
            x_mean_batch = x[idx].mean(dim=0).detach()
            z_mean_batch = z[idx].mean(dim=0).detach()
            x_var_batch = ((x[idx] - x_mean_batch) ** 2).mean(dim=0).detach()
            z_var_batch = ((z[idx] - z_mean_batch) ** 2).mean(dim=0).detach()
            xz_cov_batch = (
                ((x[idx] - x_mean_batch) * (z[idx] - z_mean_batch)).mean(dim=0).detach()
            )

            # Momentum-based update rule
            momentum = self.momentum
            self.x_mean = (
                x_mean_batch
                if self.x_mean is None
                else (1 - momentum) * self.x_mean + momentum * x_mean_batch
            )
            self.x_var = (
                x_var_batch
                if self.x_var is None
                else (1 - momentum) * self.x_var + momentum * x_var_batch
            )
            self.z_mean = (
                z_mean_batch
                if self.z_mean is None
                else (1 - momentum) * self.z_mean + momentum * z_mean_batch
            )
            self.z_var = (
                z_var_batch
                if self.z_var is None
                else (1 - momentum) * self.z_var + momentum * z_var_batch
            )
            self.xz_cov = (
                xz_cov_batch
                if self.xz_cov is None
                else (1 - momentum) * self.xz_cov + momentum * xz_cov_batch
            )

        # log r(x, z) = log p(x, z)/p(x)/p(z), with covariance given by [[x_var, xz_cov], [xz_cov, z_var]]
        x, z = swyft.equalize_tensors(x, z)
        xb = (x - self.x_mean) / self.x_var**0.5
        zb = (z - self.z_mean) / self.z_var**0.5
        rho = self.xz_cov / self.x_var**0.5 / self.z_var**0.5
        rho = torch.clip(
            rho, min=-((1 - self.minstd**2) ** 0.5), max=(1 - self.minstd**2) ** 0.5
        )
        logratios = (
            -0.5 * torch.log(1 - rho**2)
            + rho / (1 - rho**2) * xb * zb
            - 0.5 * rho**2 / (1 - rho**2) * (xb**2 + zb**2)
        )
        out = LogRatioSamples(
            logratios, z.unsqueeze(-1), self.varnames, metadata={"type": "Gaussian1d"}
        )
        return out
