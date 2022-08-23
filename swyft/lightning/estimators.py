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
        assert m%n == 0, "Cannot equalize tensors with non-divisible batch sizes."
        shape = [1 for _ in range(a.dim())]
        shape[0] = m//n
        return a.repeat(*shape), b
    else:
        assert n%m == 0, "Cannot equalize tensors with non-divisible batch sizes."
        shape = [1 for _ in range(b.dim())]
        shape[0] = n//m
        return a, b.repeat(*shape)


class LogRatioEstimator_Ndim(torch.nn.Module):
    def __init__(self, num_features, marginals, varnames = None, dropout = 0.1, hidden_features = 64, num_blocks = 2):
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
            dropout_probability = dropout,
            num_blocks=num_blocks,
        )
        if isinstance(varnames, str):
            basename = varnames
            varnames = []
            for marg in marginals:
                varnames.append([basename + "[%i]"%i for i in marg])
        self.varnames = varnames
        
    def forward(self, x, z):
        x, z = equalize_tensors(x, z)
        z = self.ptrans(z)
        ratios = self.classifier(x, z)
        w = LogRatioSamples(z, ratios, np.array(self.varnames), metadata = {"type": "MarginalMLP", "marginals": self.marginals})
        return w
    
# TODO: Introduce RatioEstimatorDense
class RatioEstimatorMLPnd(torch.nn.Module):
    def __init__(self, x_dim, marginals, dropout = 0.1, hidden_features = 64, num_blocks = 2):
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
            dropout_probability = dropout,
            num_blocks=num_blocks,
        )
        
    def forward(self, x, z):
        x, z = equalize_tensors(x, z)
        z = self.ptrans(z)
        ratios = self.classifier(x, z)
        w = LogRatioSamples(z, ratios, metadata = {"type": "MarginalMLP", "marginals": self.marginals})
        return w
    
# TODO: Deprecated class (reason: Change of name)
class RatioEstimatorMLP1d(torch.nn.Module):
    def __init__(self, x_dim, z_dim, varname = None, varnames = None, dropout = 0.1, hidden_features = 64, num_blocks = 2, use_batch_norm = True, ptrans_online_z_score = True):
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
            dropout_probability = dropout,
            num_blocks=num_blocks,
            use_batch_norm = use_batch_norm
        )
        if isinstance(varnames, list):
            self.varnames = np.array(varnames)
        else:
            self.varnames = np.array([varnames + "[%i]"%i for i in range(z_dim)])
        
    def forward(self, x, z):
        x, z = equalize_tensors(x, z)
        zt = self.ptrans(z).detach()
        logratios = self.classifier(x, zt)
        w = LogRatioSamples(z, logratios, self.varnames, metadata = {"type": "MLP1d"})
        return w


class LogRatioEstimator_1dim(torch.nn.Module):
    def __init__(self, num_features, num_params, varnames = None, dropout = 0.1, hidden_features = 64, num_blocks = 2, use_batch_norm = True, ptrans_online_z_score = True):
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
            dropout_probability = dropout,
            num_blocks=num_blocks,
            use_batch_norm = use_batch_norm
        )
        if isinstance(varnames, list):
            self.varnames = np.array(varnames)
        else:
            self.varnames = np.array([varnames + "[%i]"%i for i in range(num_params)])
        
    def forward(self, x, z):
        x, z = equalize_tensors(x, z)
        zt = self.ptrans(z).detach()
        logratios = self.classifier(x, zt)
        w = LogRatioSamples(z, logratios, self.varnames, metadata = {"type": "MLP1d"})
        return w


class RatioEstimatorGaussian1d(torch.nn.Module):
    def __init__(self, momentum = 0.1):
        super().__init__()
        self.momentum = momentum        
        self.x_mean = None
        self.z_mean = None
        self.x_var = None
        self.z_var = None
        self.xz_cov = None
        
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """2-dim Gaussian approximation to marginals and joint, assuming (B, N)."""
        print("Warning: deprecated, might be broken")
        x, z = equalize_tensors(x, z)
        if self.training or self.x_mean is None:
            # Covariance estimates must be based on joined samples only
            # NOTE: This makes assumptions about the structure of mini batches during training (J, M, M, J, J, M, M, J, ...)
            # TODO: Change to (J, M, J, M, J, M, ...) in the future
            batch_size = len(x)
            #idx = np.array([[i, i+3] for i in np.arange(0, batch_size, 4)]).flatten() 
            idx = np.arange(batch_size//2)  # TODO: Assuming (J, J, J, J, M, M, M, M) etc
            
            # Estimation w/o Bessel's correction, using simple MLE estimate (https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices)
            x_mean_batch = x[idx].mean(dim=0).detach()
            z_mean_batch = z[idx].mean(dim=0).detach()
            x_var_batch = ((x[idx]-x_mean_batch)**2).mean(dim=0).detach()
            z_var_batch = ((z[idx]-z_mean_batch)**2).mean(dim=0).detach()
            xz_cov_batch = ((x[idx]-x_mean_batch)*(z[idx]-z_mean_batch)).mean(dim=0).detach()
            
            # Momentum-based update rule
            momentum = self.momentum
            self.x_mean = x_mean_batch if self.x_mean is None else (1-momentum)*self.x_mean + momentum*x_mean_batch
            self.x_var = x_var_batch if self.x_var is None else (1-momentum)*self.x_var + momentum*x_var_batch
            self.z_mean = z_mean_batch if self.z_mean is None else (1-momentum)*self.z_mean + momentum*z_mean_batch
            self.z_var = z_var_batch if self.z_var is None else (1-momentum)*self.z_var + momentum*z_var_batch
            self.xz_cov = xz_cov_batch if self.xz_cov is None else (1-momentum)*self.xz_cov + momentum*xz_cov_batch
            
        # log r(x, z) = log p(x, z)/p(x)/p(z), with covariance given by [[x_var, xz_cov], [xz_cov, z_var]]
        xb = (x-self.x_mean)/self.x_var**0.5
        zb = (z-self.z_mean)/self.z_var**0.5
        rho = self.xz_cov/self.x_var**0.5/self.z_var**0.5
        r = -0.5*torch.log(1-rho**2) + rho/(1-rho**2)*xb*zb - 0.5*rho**2/(1-rho**2)*(xb**2 + zb**2)
        #out = torch.cat([r.unsqueeze(-1), z.unsqueeze(-1).detach()], dim=-1)
        out = LogRatioSamples(z, r, metadata = {"type": "Gaussian1d"})
        return out


