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
            Lmax=Lmax
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

    def get_z_estimate(self, x):
        z_estimator = (x - self.x_mean)*self.xz_cov/self.x_var**0.5 + self.z_mean
        return z_estimator


class LogRatioEstimator_Autoregressive(nn.Module):
    r"""Conventional autoregressive model, based on swyft.LogRatioEstimator_1dim."""
    def __init__(self, num_features, num_params, varnames, dropout = 0.1, num_blocks = 2, hidden_features = 64):
        super().__init__()
        self.cl1 = swyft.LogRatioEstimator_1dim(
            num_features = num_features + num_params,
            num_params = num_params,
            varnames = varnames,
            dropout = dropout,
            num_blocks = num_blocks,
            hidden_features = hidden_features,
            Lmax=0,
        )
        self.cl2 = swyft.LogRatioEstimator_1dim(
            num_features = num_features + num_params,
            num_params = num_params,
            varnames = varnames,
            dropout = dropout,
            num_blocks = num_blocks,
            hidden_features = hidden_features,
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
            mask[i, -self.num_params+i:] = 0
        fA = fA*mask
        logratios1 = self.cl1(fA, zB)
        
        fA = torch.cat([xA*0, zA], dim=-1)
        fA = fA.unsqueeze(1)
        fA = fA.repeat((1, self.num_params, 1))
        mask = torch.ones(self.num_params, fA.shape[-1], device=fA.device)
        for i in range(self.num_params):
            mask[i, -self.num_params+i:] = 0
        fA = fA*mask
        logratios2 = self.cl2(fA, zB)
        
        l1 = logratios1.logratios.sum(-1)
        l2 = logratios2.logratios.sum(-1)
        l2 = torch.where(l2 > 0, l2, 0)
        l = (l1-l2).detach().unsqueeze(-1)
        
        logratios_tot = swyft.LogRatioSamples(l, logratios1.params, logratios1.parnames)
                
        return dict(lrs_total=logratios_tot, lrs_partials1=logratios1, lrs_partials2=logratios2)

    
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
    def _get_mean_cov(x, correction = 1):
        # (B, *, D)
        mean = x.mean(dim=0) # (*, D)
        diffs = x - mean # (B, *, D)
        N = len(x)
        covs = torch.einsum(diffs.unsqueeze(-1), [0, ...], diffs.unsqueeze(-2), [0, ...], [...])/(N-correction)
        return mean, covs  # (*, D), (*, D, D)


    @property
    def cov(self):
        return self._cov + torch.eye(self._mean.shape[-1]).to(self._cov.device)*self._minstd**2

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
            self._mean, covariance_matrix = cov.double()) 
        logprobs_ab = dist_ab.log_prob(X)
        
        dist_b = torch.distributions.multivariate_normal.MultivariateNormal(
            self._mean[..., a_dim:], covariance_matrix = cov[..., a_dim:, a_dim:].double()) 
        logprobs_b = dist_b.log_prob(X[..., a_dim:])
        
        dist_a = torch.distributions.multivariate_normal.MultivariateNormal(
            self._mean[..., :a_dim], covariance_matrix = cov[..., :a_dim, :a_dim].double()) 
        logprobs_a = dist_a.log_prob(X[..., :a_dim])
        
        logratios = logprobs_ab - logprobs_b - logprobs_a
        
        lrs = swyft.LogRatioSamples(logratios, a, self.varnames)
           
        return lrs
    
    
class LogRatioEstimator_Autoregressive_Gaussian(nn.Module):
    r"""Estimate Gaussian log-ratios with an autoregressive model.

    Args:
        num_params: Length of parameter vector.
        varnames: List of names of parameter vector. If a single string is provided, indices are attached automatically.
        L_init: Optional initial values for L-matrix.
        minstd: Minimum standard deviation to enforce numerical stability

    The forward method returns an swyft.LogRatioSamples object.
    """

    def __init__(self, num_params, varnames, L_init = None, minstd: float = 1e-10):
        super().__init__()
        self.cl = LogRatioEstimator_Gaussian(num_params, varnames = varnames, minstd = minstd)
        self.num_params = num_params
        self._mask = nn.Parameter(self._get_mask(num_params), requires_grad = False)
        if L_init is None:
            L_init = 0.1*torch.ones(num_params, num_params)
        self.L_full = nn.Parameter(L_init, requires_grad = True)

    @property
    def L(self):
        """Lower triangular matrix of parameter correlations."""
        return self._mask*self.L_full

    @staticmethod
    def _get_mask(D):
        mask = torch.ones(D, D)
        for i in range(D):
            mask[i, i:] = 0
        return mask

    def forward(self, xA, zA, zB):
        """Forward method.

        Args:
            xA: Data vector (num_params,)
            zA: True parameters (num_params,)
            zB: Contrastive parameters (num_params,)

        Returns:
            swyft.LogRatioSamples
        """
        lA = torch.matmul(zA, self.L.T) # + torch.randn_like(xA)*1e-1
        fA = torch.stack([xA, lA], dim=-1)
        logratios = self.cl(fA, zB.unsqueeze(-1))

        return logratios

    def get_likelihood_components(self, x, double_precision = True, enforce_positivity = True):
        """Returns linear and quadratic component of likelihood ln p(x|z).

        ln p(x|z) = -1/2 * [ z.T invN z + 2 z.T b ] + const(x)

        Args:
            x: Data vector
            double_precision: Use double precision for matrix inversions
            enfore_positivity: Add diagonal matrix to enfore positivity of quadratic terms

        Returns:
            invN, b: torch.Tensor, torch.Tensor
        """
        mean = self.cl.mean.detach()
        cov = self.cl.cov.detach()
        L = self.L.detach()

        if double_precision:
            cov = cov.double()
            mean = mean.double()
            L = L.double()
            x = x.double()

        xm, lm, zm = mean.T
        invSigma_eff = torch.linalg.inv(cov)
        invSigma_eff[:,1:2,1:2] += torch.linalg.inv(cov[:,1:2,1:2])
        invSigma_eff[:,:2,:2] -= torch.linalg.inv(cov[:,:2,:2])
        invSigma_eff[:,1:,1:] -= torch.linalg.inv(cov[:,1:,1:])
        D11 = invSigma_eff[:,0,0]
        D22 = invSigma_eff[:,1,1]
        D33 = invSigma_eff[:,2,2]
        D23 = invSigma_eff[:,1,2]
        D12 = invSigma_eff[:,0,1]
        D13 = invSigma_eff[:,0,2]

        quadratic = (
            torch.diag(D33)
            + torch.matmul(torch.matmul(L.T, torch.diag(D22)), L)
            + torch.matmul(L.T, torch.diag(D23))
            + torch.matmul(torch.diag(D23), L)
        )

        # NEXT: Check if really necessary
        if enforce_positivity:
            mineig = torch.linalg.eig(quadratic).eigenvalues.real.min()
        #    print(mineig)
            quadratic = quadratic - torch.eye(len(quadratic)).to(quadratic.device)*mineig*1.0

        linear = (
        torch.matmul(torch.diag(D13)+torch.matmul(L.T, torch.diag(D12)), x-xm)-
        torch.matmul(torch.diag(D23)+torch.matmul(L.T, torch.diag(D22)), lm)-
        torch.matmul(torch.diag(D33)+torch.matmul(L.T, torch.diag(D23)), zm)
        )

        return quadratic, linear

    def get_MAP(self, x, prior_cov = None, double_precision = True):
        """Generate MAP estimator for z.

        Args:
            x: Data vector
            prior_cov: Prior covariance matrix
            double_precision: Use double precision for matrix inversion

        Returns:
            torch.tensor: MAP estimator
        """

        invN, b = self.get_likelihood_components(x, double_precision = double_precision, enforce_positivity = False)
        if prior_cov is not None:
            z_MAP = torch.matmul(torch.linalg.inv(invN + torch.linalg.inv(prior_cov)), -b)
        else:
            z_MAP = torch.matmul(torch.linalg.inv(invN), -b)
        return z_MAP

    def get_post_samples(self, N, x, prior_cov = None, gamma = 1.):
        """Generate samples for z, using standard matrix inversion (Cholesky decomposition).

        Args:
            x: Data vector
            prior_cov: Prior covariance matrix
            gamma: Rescaling factor for likelihood covariance matrix

        Returns:
            Samples

        Note: This is expected to work for significantly less than 1e4
        dimensions. For more parameters, other techniques directly based on the
        likelihood quadratic and linear components should be used.
        """
        best = self.get_MAP(x, prior_cov = prior_cov)
        invN, _ = self.get_likelihood_components(x, enforce_positivity = True)
        invN = invN*gamma
        full_cov = torch.linalg.inv(invN+torch.linalg.inv(prior_cov))
        dist = torch.distributions.MultivariateNormal(best, full_cov)
        draws = dist.sample(torch.Size([N]))
        return draws


class LogRatioEstimator_Autoregressive_Gaussian2(nn.Module):
    r"""Estimate Gaussian log-ratios with an autoregressive model.

    Args:
        num_params: Length of parameter vector.
        varnames: List of names of parameter vector. If a single string is provided, indices are attached automatically.
        L_init: Optional initial values for L-matrix.
        minstd: Minimum standard deviation to enforce numerical stability

    The forward method returns an swyft.LogRatioSamples object.
    """

    def __init__(self, num_params, varnames, L1_init = None, L2_init = None, minstd: float = 1e-10):
        super().__init__()
        self.cl1 = LogRatioEstimator_Gaussian(num_params, varnames = varnames, minstd = minstd)  # Estimate prior
        self.cl2 = LogRatioEstimator_Gaussian(num_params, varnames = varnames, minstd = minstd)  # Estimate likelihood
        self.num_params = num_params
        self._mask = nn.Parameter(self._get_mask(num_params), requires_grad = False)
        if L1_init is None:
            L1_init = torch.ones(num_params, num_params)*0.1
#            L1_init += torch.diag(torch.ones(num_params-1), diagonal=-1)*0.1
        self.L1_full = nn.Parameter(L1_init, requires_grad = True)
        if L2_init is None:
            L2_init = torch.ones(num_params, num_params)*0.1
#            L2_init += torch.diag(torch.ones(num_params-1), diagonal=-1)*0.1
        self.L2_full = nn.Parameter(L2_init, requires_grad = True)

    @property
    def L1(self):
        """Lower triangular matrix of parameter correlations."""
        return self._mask*self.L1_full

    @property
    def L2(self):
        """Lower triangular matrix of parameter correlations."""
        return self._mask*self.L2_full

    @staticmethod
    def _get_mask(D):
        "Autoregressive masking"
        mask = torch.ones(D, D)
        for i in range(D):
            mask[i, i:] = 0
        return mask

    def forward(self, xA, zA, zB):
        """Forward method.

        Args:
            xA: Data vector from A (num_params,)
            zA: Model parameters from A (num_params,)
            zB: Model parameters from B (num_params,)

        Returns:
            swyft.LogRatioSamples
        """
        L1zB = torch.matmul(zB, self.L1.T)
        logratios1 = self.cl1(zA.unsqueeze(-1), L1zB.unsqueeze(-1))  # (z; L1 z)

        G, D = self.get_prior_decomposition()
        GzA = torch.matmul(zA, G.T.detach())
        GzB = torch.matmul(zB, G.T.detach())

        L2GzA = torch.matmul(GzA, self.L2.T)
        fA = torch.stack([xA, L2GzA], dim=-1)
        logratios2 = self.cl2(fA, GzB.unsqueeze(-1))  # (x, L2 G z; G z)

        return logratios1, logratios2

    def get_prior_decomposition(self):
        """Returns estimate of the prior, such that 
            inv_Sigma = G.T * D * G
        is the inverse of the prior covariance matrix.  Here D, is diagonal and G is a lower triangular matrix.
        """
        cov = self.cl1.cov
        invCov = torch.linalg.inv(cov)
        G = torch.eye(len(self.L1)).to(cov.device) + torch.matmul(torch.diag(invCov[:,0,1]/invCov[:,0,0]), self.L1)
        D = invCov[:,0,0]
        return G, D

    def get_likelihood_components(self, x, double_precision = True, enforce_positivity = True):
        """Returns linear and quadratic component of likelihood ln p(x|z).

        ln p(x|z) = -1/2 * [ z.T invN z + 2 z.T b ] + const(x)

        Args:
            x: Data vector
            double_precision: Use double precision for matrix inversions
            enfore_positivity: Add diagonal matrix to enfore positivity of quadratic terms

        Returns:
            invN, b: torch.Tensor, torch.Tensor
        """
        mean = self.cl2.mean.detach()
        cov = self.cl2.cov.detach()
        L2 = self.L2.detach()
        G, D = self.get_prior_decomposition()
        #print(D[:10])

        if double_precision:
            cov = cov.double()
            mean = mean.double()
            L2 = L2.double()
            x = x.double()
            G = G.double()

        xm, lm, zm = mean.T
        invSigma_eff = torch.linalg.inv(cov)
        invSigma_eff[:,2:,2:] -= torch.linalg.inv(cov[:,2:,2:])
        #invSigma_eff = torch.linalg.inv(cov)
        #invSigma_eff[:,1:2,1:2] += torch.linalg.inv(cov[:,1:2,1:2])
        #invSigma_eff[:,:2,:2] -= torch.linalg.inv(cov[:,:2,:2])
        #invSigma_eff[:,1:,1:] -= torch.linalg.inv(cov[:,1:,1:])
        D00 = invSigma_eff[:,0,0]
        D11 = invSigma_eff[:,1,1]
        D22 = invSigma_eff[:,2,2]
        D12 = invSigma_eff[:,1,2]
        D01 = invSigma_eff[:,0,1]
        D02 = invSigma_eff[:,0,2]

        quadratic = (
            torch.diag(D22)
            + torch.matmul(torch.matmul(L2.T, torch.diag(D11)), L2)
            + torch.matmul(L2.T, torch.diag(D12))
            + torch.matmul(torch.diag(D12), L2)
        )
        quadratic = torch.matmul(torch.matmul(G.T, quadratic), G)

        # NEXT: Check if really necessary
#        if enforce_positivity:
#        mineig = torch.linalg.eig(quadratic).eigenvalues.real.min()
#        print(mineig)
#        quadratic = quadratic - torch.eye(len(quadratic)).to(quadratic.device)*mineig*1.0

        linear = (
        torch.matmul(torch.diag(D02)+torch.matmul(L2.T, torch.diag(D01)), x-xm)-
        torch.matmul(torch.diag(D12)+torch.matmul(L2.T, torch.diag(D11)), lm)-
        torch.matmul(torch.diag(D22)+torch.matmul(L2.T, torch.diag(D12)), zm)
        )
        linear = torch.matmul(G.T, linear)

        return quadratic, linear

    def get_MAP(self, x, prior_cov = None, double_precision = True):
        """Generate MAP estimator for z.

        Args:
            x: Data vector
            prior_cov: Prior covariance matrix
            double_precision: Use double precision for matrix inversion

        Returns:
            torch.tensor: MAP estimator
        """

        invN, b = self.get_likelihood_components(x, double_precision = double_precision, enforce_positivity = False)
        if prior_cov is not None:
            z_MAP = torch.matmul(torch.linalg.inv(invN + torch.linalg.inv(prior_cov)), -b)
        else:
            z_MAP = torch.matmul(torch.linalg.inv(invN), -b)
        return z_MAP

    def get_post_samples(self, N, x, prior_cov = None, gamma = 1.):
        """Generate samples for z, using standard matrix inversion (Cholesky decomposition).

        Args:
            x: Data vector
            prior_cov: Prior covariance matrix
            gamma: Rescaling factor for likelihood covariance matrix

        Returns:
            Samples

        Note: This is expected to work for significantly less than 1e4
        dimensions. For more parameters, other techniques directly based on the
        likelihood quadratic and linear components should be used.
        """
        best = self.get_MAP(x, prior_cov = prior_cov)
        invN, _ = self.get_likelihood_components(x, enforce_positivity = True)
        invN = invN*gamma
        full_cov = torch.linalg.inv(invN+torch.linalg.inv(prior_cov))
        dist = torch.distributions.MultivariateNormal(best, full_cov)
        draws = dist.sample(torch.Size([N]))
        return draws
