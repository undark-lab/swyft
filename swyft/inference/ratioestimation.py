# pylint: disable=no-member, not-callable
from typing import Optional

import numpy as np
import torch.nn as nn

from swyft.inference.networks import DefaultHead, DefaultTail
from swyft.inference.train import ParamDictDataset, trainloop
from swyft.nn.module import Module
from swyft.types import Array, Device, Sequence, Tuple
from swyft.utils import (
    dict_to_tensor,
    dict_to_tensor_unsqueeze,
    format_param_list,
    get_obs_shapes,
)

class SingleRatio:
    """Single ratio as function of hypercube parameters u.  Input for bound calculations."""
    def __init__(self, rc, obs, comb):
        self._rc = rc
        self._obs = obs
        self._comb = comb

    def __call__(self, u):
        ratios = self._rc.ratios(self._obs, u)
        return ratios[self._comb]



class RatioCollection:
    _save_attrs = ["param_list", "_head_swyft_state_dict", "_tail_swyft_state_dict"]

    def __init__(
        self,
        param_list,
        head: Optional[nn.Module] = DefaultHead,
        tail: Optional[nn.Module] = DefaultTail,
        head_args={},
        tail_args={},
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
        self.param_list = format_param_list(param_list)
        self.device = device

        if isinstance(head, type):
            self._uninitialized_head = head
            self._uninitialized_head_args = head_args
            self.head = None
        else:
            self.head = head
        if isinstance(head, type):
            self._uninitialized_tail = tail
            self._uninitialized_tail_args = tail_args
            self.tail = None
        else:
            self.tail = tail

        self._train_diagnostics = []

    def _init_networks(self, dataset):
        obs_shapes = get_obs_shapes(dataset[0]["obs"])
        self.head = self._uninitialized_head(
            obs_shapes, **self._uninitialized_head_args
        ).to(self.device)
        self.tail = self._uninitialized_tail(
            self.head.n_features, self.param_list, **self._uninitialized_tail_args
        ).to(self.device)

    def train(
        self,
        points,
        max_epochs: int = 10,
        batch_size: int = 32,
        lr_schedule: Sequence[float] = [1e-3, 3e-4, 1e-4],
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
        dataset = ParamDictDataset(points)

        if self.tail is None:
            self._init_networks(dataset)

        self.head.train()
        self.tail.train()

        diagnostics = trainloop(
            self.head,
            self.tail,
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
        self._train_diagnostics.append(diagnostics)

    # FIXME: Type annotations and docstring are wrong
    def ratios(
        self,
        obs: Array,
        params: Array,
        n_batch=100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve estimated marginal posterior.

        Args:
            x0: real observation to calculate posterior
            combination_indices: z indices in self.combinations

        Returns:
            parameter array, posterior array
        """

        self.head.eval()
        self.tail.eval()

        # obs = dict_to_tensor(obs, device = self.device)
        obs = dict_to_tensor_unsqueeze(obs, device=self.device)
        f = self.head(obs)

        npar = len(params[list(params)[0]])

        if npar < n_batch:
            params = dict_to_tensor(params, device=self.device)
            f = f.unsqueeze(0).expand(npar, -1)
            ratios = self.tail(f, params).detach().cpu().numpy()
        else:
            ratios = []
            for i in range(npar // n_batch + 1):
                params_batch = dict_to_tensor(
                    params,
                    device=self.device,
                    indices=slice(i * n_batch, (i + 1) * n_batch),
                )
                n = len(params_batch[list(params_batch)[0]])
                # f_batch = f.unsqueeze(0).expand(n, -1)
                f_batch = f.expand(n, -1)
                tmp = self.tail(f_batch, params_batch).detach().cpu().numpy()
                ratios.append(tmp)
            ratios = np.vstack(ratios)

        return {k: ratios[..., i] for i, k in enumerate(self.param_list)}

    @property
    def _tail_swyft_state_dict(self):
        return self.tail.swyft_state_dict()

    @property
    def _head_swyft_state_dict(self):
        return self.head.swyft_state_dict()

    def state_dict(self):
        """Return state dictionary."""
        return {attr: getattr(self, attr) for attr in RatioEstimator._save_attrs}

    @classmethod
    def from_state_dict(cls, state_dict, device: Device = "cpu"):
        """Instantiate RatioEstimator from state dictionary."""
        re = cls(state_dict["param_list"], head=None, tail=None, device=device)
        re.head = Module.from_swyft_state_dict(state_dict["_head_swyft_state_dict"]).to(
            device
        )
        re.tail = Module.from_swyft_state_dict(state_dict["_tail_swyft_state_dict"]).to(
            device
        )
        return re

#    # FIXME: Ditch posterior method, show live separately
#    def posterior(self, obs0, prior, n_samples=100000):
#        """Resturn weighted posterior samples for given observation.
#
#        Args:
#            obs0 (dict): Observation of interest.
#            prior (Prior): (Constrained) prior used to generate training data.
#            n_samples (int): Number of samples to return.
#
#        Note:
#            log_priors are not normalized.
#        """
#        pars = prior.sample(n_samples)  # prior samples
#
#        # Unmasked original wrongly normalized log_prob densities
#        log_probs = prior.log_prob(pars, unmasked=True)
#
#        ratios = self.ratios(obs0, pars)  # evaluate lnL for reference observation
#        weights = {}
#        for k, v in ratios.items():
#            weights[k] = np.exp(v)
#        return dict(params=pars, weights=weights, log_priors=log_probs)
