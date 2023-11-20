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
import itertools
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
        Lmax=0,
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
            Lmax=Lmax,
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


#class VectorPicker:
#    """(B, N) z, (marginal indices), string, flags
#
#    Args:
#        marginals: List of index tuples.
#    
#    """
#    def __init__(self, marginals, varname = None):
#        self.marginals = marginals
#        self.num_marginals = len(marginals)
#        self.dim_marginals = max([len(m) for m in marginals])
#
#        self.ptrans = swyft.networks.ParameterTransform(
#            self.num_marginals, self.marginals, online_z_score=False
#        )
#
#        self.varname = varname
#
#    def __call__(self, z):
#        """Returns (B, M, P)"""
#        return self.ptrans(z)
#
#    def varnames(self):
#        """Returns list of variable names."""


class Correlator(torch.nn.Module):
    """(B, M, S) features, (B, M, P) parameters, (M,) names, (B, M) ratios."""

    def __init__(
        self,
        num_marginals,
        num_features,
        num_params,
        varnames = None,
        dropout=0.1,
        hidden_features=64,
        num_blocks=4,
        Lmax=0,
    ):
        super().__init__()
#        print(num_marginals, num_features, num_params)
        self.classifier = swyft.networks.MarginalClassifier(
            num_marginals,
            num_features*num_params + num_params,
            hidden_features = hidden_features,
            dropout_probability=dropout,
            num_blocks=num_blocks,
            Lmax=Lmax,
        )
        self.varnames = varnames

    def forward(self, x, z):
        x, z = equalize_tensors(x, z)
 #       print(x.shape, z.shape)
        ratios = self.classifier(x, z)
 #       print(ratios.shape)
        w = LogRatioSamples(
            ratios,
            z,
            self.varnames,
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
        Lmax=0,
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
            Lmax=Lmax,
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
    """Estimating posteriors assuming that they are Gaussian.

    DEPRECATED: Use LogRatioEstimator_Gaussian instead.
    """

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
        xb = (x - self.x_mean) / self.x_var ** 0.5
        zb = (z - self.z_mean) / self.z_var ** 0.5
        rho = self.xz_cov / self.x_var ** 0.5 / self.z_var ** 0.5
        rho = torch.clip(
            rho, min=-((1 - self.minstd ** 2) ** 0.5), max=(1 - self.minstd ** 2) ** 0.5
        )
        logratios = (
            -0.5 * torch.log(1 - rho ** 2)
            + rho / (1 - rho ** 2) * xb * zb
            - 0.5 * rho ** 2 / (1 - rho ** 2) * (xb ** 2 + zb ** 2)
        )
        out = LogRatioSamples(
            logratios, z.unsqueeze(-1), self.varnames, metadata={"type": "Gaussian1d"}
        )
        return out

    def get_z_estimate(self, x):
        z_estimator = (x - self.x_mean) * self.xz_cov / self.x_var ** 0.5 + self.z_mean
        return z_estimator


class LogRatioEstimator_Autoregressive(nn.Module):
    r"""Conventional autoregressive model, based on swyft.LogRatioEstimator_1dim."""

    def __init__(
        self,
        num_features,
        num_params,
        varnames,
        dropout=0.1,
        num_blocks=2,
        hidden_features=64,
    ):
        super().__init__()
        self.cl1 = swyft.LogRatioEstimator_1dim(
            num_features=num_features + num_params,
            num_params=num_params,
            varnames=varnames,
            dropout=dropout,
            num_blocks=num_blocks,
            hidden_features=hidden_features,
            Lmax=0,
        )
        self.cl2 = swyft.LogRatioEstimator_1dim(
            num_features=num_features + num_params,
            num_params=num_params,
            varnames=varnames,
            dropout=dropout,
            num_blocks=num_blocks,
            hidden_features=hidden_features,
            Lmax=0,
        )
        self.num_params = num_params

    def forward(self, xA, zA, zB):
        xA, zB = swyft.equalize_tensors(xA, zB)
        xA, zA = swyft.equalize_tensors(xA, zA)

        fA = torch.cat([xA, zA], dim=-1)
        fA = fA.unsqueeze(1)
        fA = fA.repeat((1, self.num_params, 1))
        mask = torch.ones(self.num_params, fA.shape[-1], device=fA.device)
        for i in range(self.num_params):
            mask[i, -self.num_params + i :] = 0
        fA = fA * mask
        logratios1 = self.cl1(fA, zB)

        fA = torch.cat([xA * 0, zA], dim=-1)
        fA = fA.unsqueeze(1)
        fA = fA.repeat((1, self.num_params, 1))
        mask = torch.ones(self.num_params, fA.shape[-1], device=fA.device)
        for i in range(self.num_params):
            mask[i, -self.num_params + i :] = 0
        fA = fA * mask
        logratios2 = self.cl2(fA, zB)

        l1 = logratios1.logratios.sum(-1)
        l2 = logratios2.logratios.sum(-1)
        l2 = torch.where(l2 > 0, l2, 0)
        l = (l1 - l2).detach().unsqueeze(-1)

        logratios_tot = swyft.LogRatioSamples(l, logratios1.params, logratios1.parnames)

        return dict(
            lrs_total=logratios_tot, lrs_partials1=logratios1, lrs_partials2=logratios2
        )


class LogRatioEstimator_Gaussian(torch.nn.Module):
    """Estimating posteriors with Gaussian approximation.

    Args:
        num_params: Length of parameter vector.
        varnames: List of name of parameter vector. If a single string is provided, indices are attached automatically.
        momentum: Momentum of covariance and mean estimates
        minstd: Minimum standard deviation to enforce numerical stability
    """

    def __init__(
        self, num_params, varnames=None, momentum: float = 0.02, minstd: float = 1e-10
    ):
        super().__init__()
        self._momentum = momentum
        self._mean = None
        self._cov = None
        self._minstd = minstd

        if isinstance(varnames, list):
            self.varnames = np.array([[v] for v in varnames])
        else:
            self.varnames = np.array(
                [[varnames + "[%i]" % i] for i in range(num_params)]
            )

    @staticmethod
    def _get_mean_cov(x, correction=1):
        # (B, *, D)
        mean = x.mean(dim=0)  # (*, D)
        diffs = x - mean  # (B, *, D)
        N = len(x)
        covs = torch.einsum(
            diffs.unsqueeze(-1), [0, ...], diffs.unsqueeze(-2), [0, ...], [...]
        ) / (N - correction)
        return mean, covs  # (*, D), (*, D, D)

    @property
    def cov(self):
        return (
            self._cov
            + torch.eye(self._mean.shape[-1]).to(self._cov.device) * self._minstd ** 2
        )

    @property
    def mean(self):
        return self._mean

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        """Gaussian approximation to marginals and joint, assuming (B, N).

        a shape: (B, N, D1)
        b shape: (B, N, D2)

        """

        a_dim = a.shape[-1]
        b_dim = b.shape[-1]

        if self.training or self._mean is None:
            batch_size = len(a)
            idx = np.arange(batch_size)

            X = torch.cat([a[idx], b[idx]], dim=-1).detach()

            # Estimation w/o Bessel's correction
            # Using simple MLE estimate (https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices)
            mean_batch, cov_batch = self._get_mean_cov(X, correction=0)

            # Momentum-based update rule
            momentum = self._momentum
            self._mean = (
                mean_batch
                if self._mean is None
                else (1 - momentum) * self._mean + momentum * mean_batch
            )
            self._cov = (
                cov_batch
                if self._cov is None
                else (1 - momentum) * self._cov + momentum * cov_batch
            )

        cov = self.cov

        # Match tensor batch dimensions
        a, b = swyft.equalize_tensors(a, b)

        # Get standard normal distributed parameters
        X = torch.cat([a, b], dim=-1).double()

        dist_ab = torch.distributions.multivariate_normal.MultivariateNormal(
            self._mean, covariance_matrix=cov.double()
        )
        logprobs_ab = dist_ab.log_prob(X)

        dist_b = torch.distributions.multivariate_normal.MultivariateNormal(
            self._mean[..., a_dim:], covariance_matrix=cov[..., a_dim:, a_dim:].double()
        )
        logprobs_b = dist_b.log_prob(X[..., a_dim:])

        dist_a = torch.distributions.multivariate_normal.MultivariateNormal(
            self._mean[..., :a_dim], covariance_matrix=cov[..., :a_dim, :a_dim].double()
        )
        logprobs_a = dist_a.log_prob(X[..., :a_dim])

        logratios = logprobs_ab - logprobs_b - logprobs_a

        lrs = swyft.LogRatioSamples(logratios, a, self.varnames)

        return lrs

    
def get_marginal_index_combinations(marginals, projection):
    """Returns a list of marginal index combinations.
    
    Args:
        marginals: int or list of marginals to consider
        projection: '1d', '2d', '3d', 'joined', or combination thereof
        
    Returns:
        Tuple of tuples
    """
    if isinstance(marginals, int):
        marginals = range(marginals)
    out = []  
    if '1d' in projection:
        for i in range(len(marginals)):
            out.append((marginals[i],))
    if '2d' in projection:
        for i in range(len(marginals)):
            for j in range(i+1, len(marginals)):
                out.append((marginals[i], marginals[j]))
    if '3d' in projection:
        for i in range(len(marginals)):
            for j in range(i+1, len(marginals)):
                for k in range(j+1, len(marginals)):
                    out.append((marginals[i], marginals[j], marginals[k]))
    if 'joined' in projection:
        joined = tuple(marginals)
        if not joined in out:
            out.append(joined)
    return tuple(out)


class MarginalProjector:
    """Marginal projection of parameters and data summaries.
    
    Args:
        marginals: list of marginals (tuple of tuples)
        varname: Variable name (str)
    """
    def __init__(self, marginals, varname = None):
        self._marginals = marginals
        self._compressed_marginals, self._num_unique_indices = self._get_compressed_marginals(marginals)
        
        self._padded_marginals = self._pad_marginal_indices(self._marginals)
        self._padded_compressed_marginals = self._pad_marginal_indices(self._compressed_marginals)

        self._padding_mask = torch.tensor((self._padded_marginals != -1)*1)
        self._varnames = self._gen_varnames(marginals, varname)
        
    @property
    def num_marginals(self):
        """Number of marginals that will be returned"""
        return len(self._padded_marginals)
    
    @property
    def num_params(self):
        """Maximum dimension of marginal posteriors"""
        return self._padded_marginals.shape[1]
        
    @property
    def marginals(self):
        """Return list of marginals"""
        return self._marginals
        
    @property
    def varnames(self):
        return self._varnames
        
    @staticmethod
    def _gen_varnames(marginals, varname):
        if varname is None:
            return None
        v = []
        for m in marginals:
            v.append(tuple(varname+"["+str(i)+"]" for i in m))
        return tuple(v)
    
    @staticmethod
    def _get_compressed_marginals(marginals):
        # Map marginal list on list with minimum unique index range
        sorted_unique_indices = tuple(sorted(list(set(itertools.chain.from_iterable(marginals)))))
        inverse = np.arange(sorted_unique_indices[-1]+1)
        for i in range(len(sorted_unique_indices)):
            inverse[sorted_unique_indices[i]] = i
        num_unique_indices = len(sorted_unique_indices)
        compressed_marginals = tuple([tuple(inverse[i] for i in m) for m in marginals])
        return compressed_marginals, num_unique_indices

    @staticmethod
    def _pad_marginal_indices(marginals):
        # In the case of marignals with variable dimensionality, pad -1 indices
        # Those -1 components will be masked in the end
        d = max([len(m) for m in marginals])
        m_padded = [m + (-1,)*(d-len(m)) for m in marginals]
        return np.array(m_padded)
    
    def project_parameters(self, z):
        """Project parameters."""
        zp = z[..., self._padded_marginals]*self._padding_mask.to(z.device)
        return zp
    
    def project_summaries(self, s):
        """Project data summaries."""
        assert s.shape[1] >= self._num_unique_indices, "Data summary must be of form (B, U, S), with U >= %i"%self._num_unique_indices
        s = s[..., self._padded_compressed_marginals, :]*self._padding_mask.unsqueeze(-1).to(s.device)
        s = s.flatten(start_dim=-2)
        return s
    
    
class LogRatioEstimator(torch.nn.Module):
    def __init__(self, marginals, projection = '1d', varname = None, num_features = 2):
        super().__init__()
        self.marginal_indices = get_marginal_index_combinations(marginals, projection)
        self.proj = MarginalProjector(self.marginal_indices, varname = varname)
        num_marginals = self.proj.num_marginals
        num_params = self.proj.num_params
        self.corr = swyft.lightning.estimators.Correlator(num_marginals, num_features, num_params, varnames = self.proj.varnames)
        
    def forward(self, x, z):
        s = self.proj.project_summaries(x)
        z =self.proj.project_parameters(z)
        return self.corr(s, z)

