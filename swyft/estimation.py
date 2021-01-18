# pylint: disable=no-member, not-callable
from warnings import warn
from copy import deepcopy
from scipy.special import xlogy

import numpy as np
from scipy.integrate import trapz

import torch
import torch.nn as nn

from .utils import Module, get_obs_shapes

from .train import trainloop
from .cache import Dataset, Normalize
from .network import DefaultTail, DefaultHead
from .types import (
    Sequence,
    Tuple,
    Device,
    Combinations,
    Callable,
    Array,
    Union,
    PathType,
    Dict,
    Optional,
)
from .utils import array_to_tensor, tobytes, process_combinations, dict_to_device, dict_to_tensor

class RatioEstimator:
    _save_attrs = ["param_list", "_head_swyft_state_dict", "_tail_swyft_state_dict"]

    def __init__(
        self,
        param_list,
        head: Optional[nn.Module] = DefaultHead,
        tail: Optional[nn.Module] = DefaultTail,
        head_args = {},
        tail_args = {},
        device: Device = "cpu",
    ):
        """RatioEstimator takes simulated points from the iP3 sample cache and handles training and posterior calculation.

        Args:
            points: points dataset from the iP3 sample cache
            combinations: which combinations of z parameters to learn
            head: initialized module which processes observations, head(x0) = y
            previous_ratio_estimator: ratio estimator from another round. if given, reuse head.
            device: default is cpu
            statistics: x_mean, x_std, z_mean, z_std
        """
        self.param_list = self._format_param_list(param_list)
        self.device = device

        if type(head) == type:
            self._uninitialized_head = [head, head_args]
            self.head = None
        else:
            self.head = head
        if type(tail) == type:
            self._uninitialized_tail = [tail, tail_args]
            self.tail = None
        else:
            self.tail = tail

    def _format_param_list(self, param_list):
        out = []
        for v in param_list:
            if not isinstance(v, tuple):
                v = (v,)
            else:
                v = tuple(sorted(v))
            out.append(v)
        return out

    def _init_networks(self, dataset):
        obs_shapes = get_obs_shapes(dataset[0]['obs'])
        self.head = self._uninitialized_head[0](obs_shapes, **self._uninitialized_head[1]).to(self.device)
        self.tail = self._uninitialized_tail[0](self.head.n_features, self.param_list, **self._uninitialized_tail[1]).to(self.device)
        print("n_features =", self.head.n_features)

    def train(
        self,
        points,
        max_epochs: int = 10,
        batch_size: int = 4,
        lr_schedule: Sequence[float] = [1e-3, 1e-4],
        early_stopping_patience: int = 1,
        nworkers: int = 0,
        percent_validation=0.1,
    ) -> None:
        """Train higher-dimensional marginal posteriors.

        Args:
            max_epochs: maximum number of training epochs
            batch_size: minibatch size
            lr_schedule: list of learning rates
            early_stopping_patience: early stopping patience
            nworkers: number of Dataloader workers (0 for no dataloader parallelization)
            percent_validation: percentage to allocate to validation set
        """
        dataset = Dataset(points)

        if self.tail is None: self._init_networks(dataset)

        self.head.train()
        self.tail.train()

        trainloop(
            self.head, self.tail,
            dataset,
            combinations=None,
            device=self.device,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr_schedule=lr_schedule,
            early_stopping_patience=early_stopping_patience,
            nworkers=nworkers,
            percent_validation=percent_validation,
        )
        return None

    def lnL(
        self,
        obs: Array,
        params: Array,
        n_batch = 100,
        max_n_points: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve estimated marginal posterior.

        Args:
            x0: real observation to calculate posterior
            combination_indices: z indices in self.combinations
            max_n_points: number of points to calculate ratios on

        Returns:
            parameter array, posterior array
        """

        self.head.eval()
        self.tail.eval()

        obs = dict_to_tensor(obs, device = self.device)
        f = self.head(obs)

        npar = len(params[list(params)[0]])

        if npar < n_batch:
            params = dict_to_tensor(params, device = self.device)
            f = f.unsqueeze(0).expand(npar, -1)
            lnL = self.tail(f, params).detach().cpu().numpy()
        else:
            lnL = []
            for i in range(npar//n_batch + 1):
                params_batch = dict_to_tensor(params, device = self.device, indices = slice(i*n_batch, (i+1)*n_batch))
                n = len(params_batch[list(params_batch)[0]])
                f_batch = f.unsqueeze(0).expand(n, -1)
                tmp = self.tail(f_batch, params_batch).detach().cpu().numpy()
                lnL.append(tmp)
            lnL = np.vstack(lnL)

        return {k: lnL[..., i] for i, k in enumerate(self.param_list)}

    @property
    def _tail_swyft_state_dict(self):
        return self.tail.swyft_state_dict()

    @property
    def _head_swyft_state_dict(self):
        return self.head.swyft_state_dict()

    def state_dict(self):
        return {attr: getattr(self, attr) for attr in RatioEstimator._save_attrs}

    @classmethod
    def from_state_dict(cls, state_dict, device: Device = "cpu"):
        re = cls(state_dict['param_list'], head = None, tail = None, device = device)
        re.head = Module.from_swyft_state_dict(state_dict["_head_swyft_state_dict"]).to(device)
        re.tail = Module.from_swyft_state_dict(state_dict["_tail_swyft_state_dict"]).to(device)
        return re

    def posterior(self, obs0, prior, n_samples = 100000):
        pars = prior.sample(n_samples)  # prior samples
        lnL = self.lnL(obs0, pars)  # evaluate lnL for reference observation
        weights = {}
        for k, v in lnL.items():
            weights[k] = np.exp(v)
        return dict(params = pars, weights = weights)

        #result = {}
        #for tag in lnL.keys():
        #    #assert len(tag) == 1, "Only works for 1-dim posteriors right now"
        #    
        #    # Get ratio and its integral
        #    # r = p(z|x)/p_c(z) # posterior vs constrained prior ratio
        #    r = np.exp(lnL[tag])
        #    #r_int = sum(r)/len(r)  # approximate r integral over p_c(z)
        #    
        #    ## Generate histogram
        #    z = np.stack([pars[t] for t in tag]).T
        #    #values, z_edges = np.histogram(z, weights = r, bins = n_bins, density = True)
        #    #dz = z_edges[1:] - z_edges[:-1]
        #    #z_bar = (z_edges[1:] + z_edges[:-1])/2
        #    
        #    ## Estimate entropy
        #    #entropy = sum(xlogy(dz*values, values))

        #    #result[tag] = dict(entropy=entropy, r_int=r_int, z=z, weights=r)
        #    result[tag] = dict(z=z, weights=r)
        #
        #if verbose:
        #    for tag in result:
        #        print("Posterior:", tag)
        #        print("  Entropy  = %.3g"%result[tag]['entropy'])
        #        print("  Integral = %.3g"%result[tag]['r_int'])
        
        return result


class Points:
    """Points references (observation, parameter) pairs drawn from an inhomogenous Poisson Point Proccess (iP3) Cache.
    Points implements this via a list of indices corresponding to data contained in a cache which is provided at initialization.
    """

    _save_attrs = ["indices"]

    def __init__(self, indices, cache: "swyft.cache.Cache", noisehook=None):
        """Create a points dataset

        Args:
            cache (Cache): iP3 cache for zarr storage
            intensity (Intensity): inhomogenous Poisson Point Proccess intensity function on parameters
            noisehook (function): (optional) maps from (x, z) to x with noise
        """
        if cache.requires_sim():
            raise RuntimeError(
                "The cache has parameters without a corresponding observation. Try running the simulator."
            )

        self.cache = cache
        self.noisehook = noisehook
        self.indices = np.array(indices)

    def __len__(self):
        return len(self.indices)

    def get_range(self, indices):
        N = len(indices)
        obs_comb = {k: np.empty((N,)+v.shape) for k, v in self[0]['obs'].items()}
        par_comb = {k: np.empty((N,)+v.shape) for k, v in self[0]['par'].items()}
        
        for i in indices:
            p = self[i]
            for k,v in p['obs'].items():
                obs_comb[k][i] = v
            for k,v in p['par'].items():
                par_comb[k][i] = v

        return dict(obs=obs_comb, par=par_comb)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x_keys = list(self.cache.x)
        z_keys = list(self.cache.z)
        x = {k: self.cache.x[k][i] for k in x_keys}
        z = {k: self.cache.z[k][i] for k in z_keys}

        if self.noisehook is not None:
            x = self.noisehook(x, z)

        return dict(obs=x, par=z)
