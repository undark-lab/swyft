import logging
from warnings import warn

import numpy as np

from .posteriors import Posteriors
from swyft.utils import tupelize_marginals
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

        self.simulate = self.dataset.simulate
        self.add = self.posteriors.add
        self.train = self.posteriors.train
        self.sample = self.posteriors.sample

    def truncate(self, partition, obs0):
        partition = tupelize_marginals(partition)
        bound = swyft.Bound.from_Posteriors(partition, self.posteriors, obs0)
        return bound
