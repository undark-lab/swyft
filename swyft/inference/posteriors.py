import logging
from warnings import warn

import torch
import numpy as np

from .ratios import RatioEstimator
from swyft.networks import DefaultHead, DefaultTail
from swyft.types import Array, Device, Sequence, Tuple
import swyft

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

#    def __init__(self, ratio_collections):
#        self._rcs = ratio_collections
#        self.param_list = []
#        [self.param_list.extend(rc.param_list) for rc in self._rcs]
#        self.param_list = list(set(self.param_list))

class Posteriors:
    def __init__(self, dataset, simhook=None):
        # Store relevant information about dataset
        self._prior = dataset.prior
        self._indices = dataset.indices
        self._N = len(dataset)
        self._ratios = []

        # Temporary
        self._dataset = dataset

    def infer(
        self,
        partitions,
        train_args: dict = {},
        head=DefaultHead,
        tail=DefaultTail,
        head_args: dict = {},
        tail_args: dict = {},
        device="cpu",
    ):
        """Perform 1-dim marginal focus fits.

        Args:
            train_args (dict): Training keyword arguments.
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
        """
        ntrain = self._N
        bp = self._prior.bound

        re = self._train(
            bp,
            partitions,
            head=head,
            tail=tail,
            head_args=head_args,
            tail_args=tail_args,
            train_args=train_args,
            N=ntrain,
            device=device,
        )
        self._ratios.append(re)

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
        for rc in self._ratios:
            ratios = rc.ratios(obs, params, n_batch=n_batch)
            result.update(ratios)
        return result

    def _train(
        self, prior, param_list, N, train_args, head, tail, head_args, tail_args, device
    ):
        if param_list is None:
            param_list = prior.params()

        re = RatioEstimator(
            param_list,
            device=device,
            head=head,
            tail=tail,
            tail_args=tail_args,
            head_args=head_args,
        )
        re.train(self._dataset, **train_args)

        return re

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
