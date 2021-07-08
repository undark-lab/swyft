# pylint: disable=no-member, not-callable
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from swyft.inference.train import trainloop
from swyft.networks import DefaultHead, DefaultTail, Module
from swyft.types import Array, Device
from swyft.utils import (
    array_to_tensor,
    dict_to_tensor_unsqueeze,
    format_param_list,
    get_obs_shapes,
)


class IsolatedRatio:
    """Single ratio as function of hypercube parameters u.  Input for bound calculations."""

    def __init__(self, rc, obs, comb, zdim):
        self._rc = rc
        self._obs = obs
        self._comb = comb
        self._zdim = zdim

    def __call__(self, u, n_batch=10_000):
        U = np.random.rand(len(u), self._zdim)
        U[:, np.array(self._comb)] = u
        ratios = self._rc.ratios(self._obs, U, n_batch=n_batch)
        return ratios[self._comb]


# Deprecated
# class JoinedRatioEstimator:
#    def __init__(self, ratio_collections):
#        self._rcs = ratio_collections
#        self.param_list = []
#        [self.param_list.extend(rc.param_list) for rc in self._rcs]
#        self.param_list = list(set(self.param_list))
#
#    def ratios(self, obs: Array, params: Array, n_batch=100):
#        result = {}
#        for rc in self._rcs:
#            ratios = rc.ratios(obs, params, n_batch=n_batch)
#            result.update(ratios)
#        return result


# Deprecated
# class JoinedRatioCollection:
#    def __init__(self, ratio_collections):
#        self._rcs = ratio_collections
#        self.param_list = []
#        [self.param_list.extend(rc.param_list) for rc in self._rcs]
#        self.param_list = list(set(self.param_list))
#
#    def ratios(self, obs: Array, params: Array, device=None, n_batch=10_000):
#        result = {}
#        for rc in self._rcs:
#            ratios = rc.ratios(obs, params, device=device, n_batch=n_batch)
#            result.update(ratios)
#        return result

# class RatioCollection:


class RatioEstimator:
    _save_attrs = [
        "param_list",
        "_head_swyft_state_dict",
        "_tail_swyft_state_dict",
        "_train_diagnostics",
    ]

    def __init__(
        self,
        param_list,
        head: Optional[nn.Module] = DefaultHead,
        tail: Optional[nn.Module] = DefaultTail,
        head_args={},
        tail_args={},
        device: Device = "cpu",
    ):
        """RatioEstimator takes simulated points from the iP3 sample store and handles training and posterior calculation.

        Args:
            points: points dataset from the iP3 sample store
            head: initialized module which processes observations, head(x0) = y
            previous_ratio_estimator: ratio estimator from another round. if given, reuse head.
            device: default is cpu
            statistics: x_mean, x_std, z_mean, z_std
        """
        self.param_list = format_param_list(param_list)
        self._device = device

        if isinstance(head, type):
            self._uninitialized_head = head
            self._uninitialized_head_args = head_args
            self.head = None
        else:
            self.head = deepcopy(head).to(device)
        if isinstance(head, type):
            self._uninitialized_tail = tail
            self._uninitialized_tail_args = tail_args
            self.tail = None
        else:
            self.tail = deepcopy(tail).to(device)

        self._train_diagnostics = []

    @property
    def device(self):
        return self._device

    def _init_networks(self, dataset):
        if self.head is None:
            obs_shapes = get_obs_shapes(dataset[0][0])
            self.head = self._uninitialized_head(
                obs_shapes, **self._uninitialized_head_args
            ).to(self.device)
        if self.tail is None:
            self.tail = self._uninitialized_tail(
                self.head.n_features, self.param_list, **self._uninitialized_tail_args
            ).to(self.device)

    def to(self, device):
        self.head = self.head.to(device)
        self.tail = self.tail.to(device)
        self._device = device
        return self

    def train(
        self,
        dataset,
        batch_size=64,
        validation_size=0.1,
        early_stopping_patience=5,
        max_epochs=30,
        optimizer=torch.optim.Adam,
        optimizer_args=dict(lr=1e-3),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args=dict(reduce_lr_factor=0.1, reduce_lr_patience=5),
        nworkers=2,
        non_blocking=True,
    ) -> None:
        """Train higher-dimensional marginal posteriors.

        Args:
            nworkers: number of Dataloader workers (0 for no dataloader parallelization)
        """

        self._init_networks(dataset)
        self.head.train()
        self.tail.train()

        diagnostics = trainloop(
            head=self.head,
            tail=self.tail,
            dataset=dataset,
            batch_size=batch_size,
            validation_size=validation_size,
            early_stopping_patience=early_stopping_patience,
            max_epochs=max_epochs,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            nworkers=nworkers,
            device=self.device,
            non_blocking=non_blocking
        )
        self._train_diagnostics.append(diagnostics)

    def train_diagnostics(self):
        return self._train_diagnostics

    def ratios(self, obs, params, device=None, n_batch=10_000):
        """Retrieve estimated marginal posterior."""
        self.head.eval()
        self.tail.eval()

        # FIXME: Is this device functionality really necessary?  We can use
        # ".to()" instead

        if device is None:
            device = torch.device(self.device)
        else:
            device = torch.device(device)

        if device != self.device:
            head = deepcopy(self.head).to(device=device)
            tail = deepcopy(self.tail).to(device=device)
        else:
            head = self.head
            tail = self.tail

        with torch.no_grad():
            # obs = dict_to_tensor(obs, device = device)
            obs = dict_to_tensor_unsqueeze(obs, device=device)
            f = head(obs)

            npar = len(params)
            if npar < n_batch:
                params = array_to_tensor(params, device=device)
                f = f.expand(npar, -1)
                ratios = tail(f, params).detach().cpu().numpy()
            else:
                ratios = []
                for i in range(npar // n_batch + 1):
                    params_batch = array_to_tensor(
                        params[i * n_batch : (i + 1) * n_batch, :], device=device
                    )
                    n = len(params_batch)
                    f_batch = f.expand(n, -1)
                    tmp = tail(f_batch, params_batch).detach().cpu().numpy()
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
        """Instantiate RatioCollectoin from state dictionary."""
        head = Module.from_swyft_state_dict(state_dict["_head_swyft_state_dict"])
        tail = Module.from_swyft_state_dict(state_dict["_tail_swyft_state_dict"])
        re = cls(state_dict["param_list"], head=head, tail=tail, device=device)
        re._train_diagnostics = state_dict["_train_diagnostics"]
        return re
