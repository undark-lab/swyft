# pylint: disable=no-member, not-callable
from copy import deepcopy
from typing import Callable

import numpy as np
import torch

from swyft.inference.train import trainloop
from swyft.networks import DefaultHead, DefaultTail, Module
from swyft.types import Array, Device, MarginalsType, ObsType, RatiosType
from swyft.utils import (
    array_to_tensor,
    dict_to_tensor_unsqueeze,
    get_obs_shapes,
    tupleize_marginals,
)


class RatioEstimator:
    """RatioEstimator takes simulated points from the iP3 sample store and handles training and posterior calculation.

    Args:
        points: points dataset from the iP3 sample store
        head: initialized module which processes observations, head(x0) = y
        previous_ratio_estimator: ratio estimator from another round. if given, reuse head.
        device: default is cpu
        statistics: x_mean, x_std, z_mean, z_std
    """

    _save_attrs = [
        "marginals",
        "_head_swyft_state_dict",
        "_tail_swyft_state_dict",
        "_train_diagnostics",
    ]

    def __init__(
        self,
        marginals: MarginalsType,
        head: Callable[..., "swyft.Module"] = DefaultHead,
        tail: Callable[..., "swyft.Module"] = DefaultTail,
        head_args: dict = {},
        tail_args: dict = {},
        device: Device = "cpu",
    ) -> None:
        self.marginals = tupleize_marginals(marginals)
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
    def device(self) -> Device:
        return self._device

    def _init_networks(self, dataset: "swyft.Dataset") -> None:
        if self.head is None:
            obs_shapes = get_obs_shapes(dataset[0][0])
            self.head = self._uninitialized_head(
                obs_shapes, **self._uninitialized_head_args
            ).to(self.device)
        if self.tail is None:
            self.tail = self._uninitialized_tail(
                self.head.n_features, self.marginals, **self._uninitialized_tail_args
            ).to(self.device)

    def to(self, device: Device) -> "RatioEstimator":
        self.head = self.head.to(device)
        self.tail = self.tail.to(device)
        self._device = device
        return self

    def train(
        self, dataset: "swyft.Dataset", trainoptions: "swyft.inference.TrainOptions"
    ) -> None:
        """Train higher-dimensional marginal posteriors.

        Args:
            dataset (swyft.Dataset): Training dataset
            trainoptions (swyft.TrainOptions): swyft TrainOptions dataclass
        """

        self._init_networks(dataset)
        self.head.train()
        self.tail.train()

        diagnostics = trainloop(
            head=self.head,
            tail=self.tail,
            dataset=dataset,
            trainoptions=trainoptions,
            device=self.device,
        )
        self._train_diagnostics.append(diagnostics)

    def train_diagnostics(self):  # TODO add type annotation
        return self._train_diagnostics

    def ratios(self, obs: ObsType, params: Array, n_batch: int = 10_000) -> RatiosType:
        """Retrieve estimated marginal posterior."""
        self.head.eval()
        self.tail.eval()

        head = self.head
        tail = self.tail

        with torch.no_grad():
            # obs = dict_to_tensor(obs, device = self.device)
            obs = dict_to_tensor_unsqueeze(obs, device=self.device)
            f = head(obs)

            npar = len(params)
            if npar < n_batch:
                params = array_to_tensor(params, device=self.device)
                f = f.expand(npar, -1)
                ratios = tail(f, params).detach().cpu().numpy()
            else:
                ratios = []
                for i in range(npar // n_batch + 1):
                    params_batch = array_to_tensor(
                        params[i * n_batch : (i + 1) * n_batch, :], device=self.device
                    )
                    n = len(params_batch)
                    f_batch = f.expand(n, -1)
                    tmp = tail(f_batch, params_batch).detach().cpu().numpy()
                    ratios.append(tmp)
                ratios = np.vstack(ratios)

            return {k: ratios[..., i] for i, k in enumerate(self.marginals)}

    @property
    def _tail_swyft_state_dict(self) -> dict:
        return self.tail.swyft_state_dict()

    @property
    def _head_swyft_state_dict(self) -> dict:
        return self.head.swyft_state_dict()

    def state_dict(self) -> dict:
        """Return state dictionary."""
        return {attr: getattr(self, attr) for attr in RatioEstimator._save_attrs}

    @classmethod
    def from_state_dict(
        cls, state_dict: dict, device: Device = "cpu"
    ) -> "RatioEstimator":
        """Instantiate RatioCollection from state dictionary."""
        head = Module.from_swyft_state_dict(state_dict["_head_swyft_state_dict"])
        tail = Module.from_swyft_state_dict(state_dict["_tail_swyft_state_dict"])
        re = cls(state_dict["marginals"], head=head, tail=tail, device=device)
        re._train_diagnostics = state_dict["_train_diagnostics"]
        return re
