import logging
from warnings import warn

import numpy as np
import torch

import swyft
from swyft.types import Array
from swyft.networks import DefaultHead, DefaultTail
from swyft.utils import tupelize_marginals
from .ratios import RatioEstimator

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

class Posteriors:
    def __init__(self, dataset, simhook=None):
        # Store relevant information about dataset
        self._prior = dataset.prior
        self._indices = dataset.indices
        self._N = len(dataset)
        self._ratios = {}

        # Temporary
        self._dataset = dataset

    def add(
        self,
        marginals,
        head=DefaultHead,
        tail=DefaultTail,
        head_args: dict = {},
        tail_args: dict = {},
        device="cpu",
    ):
        """Add marginals.

        Args:
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
        """
        marginals = tupelize_marginals(marginals)
        re = RatioEstimator(
            marginals,
            device=device,
            head=head,
            tail=tail,
            head_args=head_args,
            tail_args=tail_args,
        )
        self._ratios[marginals] = re

    def train(
        self,
        marginals,
        train_args: dict = {}
    ):
        """Train marginals.

        Args:
            train_args (dict): Training keyword arguments.
        """
        marginals = tupelize_marginals(marginals)
        re = self._ratios[marginals]
        re.train(self._dataset, **train_args)

    def sample(self, N, obs0):
        """Resturn weighted posterior samples for given observation.

        Args:
            obs0 (dict): Observation of interest.
            N (int): Number of samples to return.
        """
        v = self._prior.sample(N)  # prior samples

        # Unmasked original wrongly normalized log_prob densities
        #log_probs = self._prior.log_prob(v)
        u = self._prior.ptrans.u(v)

        ratios = self._eval_ratios(obs0, u)  # evaluate lnL for reference observation
        weights = {}
        for k, val in ratios.items():
            weights[k] = np.exp(val)
        return dict(params=v, weights=weights)

    @property
    def bound(self):
        return self._prior.bound

    @property
    def ptrans(self):
        return self._prior.ptrans

    def _eval_ratios(self, obs: Array, params: Array, n_batch=100):
        result = {}
        for marginals, rc in self._ratios.items():
            ratios = rc.ratios(obs, params, n_batch=n_batch)
            result.update(ratios)
        return result

    def state_dict(self):
        state_dict = dict(
            prior=self._prior.state_dict(),
            indices=self._indices,
            N=self._N,
            ratios=[r.state_dict() for r in self._ratios],
        )
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict, dataset=None, device="cpu"):
        obj = Posteriors.__new__(Posteriors)
        obj._prior = swyft.Prior.from_state_dict(state_dict["prior"])
        obj._indices = state_dict["indices"]
        obj._N = state_dict["N"]
        obj._ratios = [
            RatioEstimator.from_state_dict(sd) for sd in state_dict["ratios"]
        ]

        obj._dataset = dataset
        obj._device = device
        return obj

    @classmethod
    def load(cls, filename, dataset=None, device="cpu"):
        sd = torch.load(filename)
        return cls.from_state_dict(sd, dataset=dataset, device=device)

    def save(self, filename):
        sd = self.state_dict()
        torch.save(sd, filename)

    @classmethod
    def from_Microscope(cls, micro):
        # FIXME: Return copy
        return micro._posteriors[-1]
