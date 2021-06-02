import logging
from warnings import warn

import numpy as np

from .posteriors import Posteriors
from swyft.networks import DefaultHead, DefaultTail
from swyft.utils import all_finite, format_param_list
import swyft

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

class Task:
    """Main SWYFT interface class."""

    def __init__(
        self,
        N,
        prior,
        store,
        simhook=None,
        device="cpu",
        bound=None
    ):
        """Initialize swyft.

        Args:
            N (int): Training points.
            prior (Prior): Prior model.
            store (Store): Storage for simulator results.  If none, create MemoryStore.
            simhook (function): Noise model, optional.
            device (str): Device.
            bound (Bound): Optional bound object.
        """
        truncated_prior = prior.rebounded(bound)
        self.dataset = swyft.Dataset(N, truncated_prior, store, simhook = simhook)
        self.posteriors = Posteriors(self.dataset)

        self._device = device

    def simulate(self):
        self.dataset.simulate()
    
    def infer(self,
        partition,
        train_args: dict = {},
        head=DefaultHead,
        tail=DefaultTail,
        head_args: dict = {},
        tail_args: dict = {},
        device="cpu",
        ):
        self.posteriors.infer(partition, train_args = train_args, head = head,
                tail = tail, head_args = head_args, tail_args = tail_args,
                device = device)

    def truncate(self, partition, obs0):
        bound = swyft.Bound.from_Posteriors(partition, self.posteriors, obs0)
        return bound

    def sample(self, N, obs0):
        return self.posteriors.sample(N, obs0)
