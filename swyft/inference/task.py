import logging
from warnings import warn

import numpy as np
import torch

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
        bound=None
    ):
        """Initialize swyft.

        Args:
            N (int): Training points.
            prior (Prior): Prior model.
            store (Store): Storage for simulator results.  If none, create MemoryStore.
            simhook (function): Noise model, optional.
            bound (Bound): Optional bound object.
        """
        truncated_prior = prior.rebounded(bound)
        dataset = swyft.Dataset(N, truncated_prior, store, simhook = simhook)
        posteriors = Posteriors(truncated_prior)
        self.from_dataset_and_posteriors(dataset, posteriors)

        self.add = self.posteriors.add
        self.sample = self.posteriors.sample

    def from_dataset_and_posteriors(self, dataset, posteriors):
        self.dataset = dataset
        self.posteriors = posteriors

    def simulate(self):
        if self.dataset is not None:
            self.dataset.simulate()
        else:
            print("WARNING: No dataset specified.")
            return

    def train(self, marginals, train_args = {}):
        if self.dataset is not None:
            self.posteriors.train(marginals, self.dataset, train_args = train_args)
        else:
            print("WARNING: No dataset specified. Cannot train")
            return

    def truncate(self, partition, obs0):
        partition = tupelize_marginals(partition)
        bound = swyft.Bound.from_Posteriors(partition, self.posteriors, obs0)
        print("Bounds: Truncating...")
        print("Bounds: ...done. New volue is V=%.4g"%bound.volume)
        return bound

    def state_dict(self):
        sd_dataset = None if self.dataset is None else self.dataset.state_dict()
        state_dict = dict(
            dataset=sd_dataset,
            posteriors=self.posteriors.state_dict(),
        )
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict, store = None):
        obj = Task.__new__(Task)
        if state_dict['dataset'] is not None and store is not None:
            dataset = swyft.Dataset.from_state_dict(state_dict['dataset'], store)
        else:
            dataset = None
        posteriors = swyft.Posteriors.from_state_dict(state_dict['posteriors'])
        obj.from_dataset_and_posteriors(dataset, posteriors)
        return obj

    @classmethod
    def load(cls, filename, store = None):
        sd = torch.load(filename)
        return cls.from_state_dict(sd, store = store)

    def save(self, filename):
        sd = self.state_dict()
        torch.save(sd, filename)
