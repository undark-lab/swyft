import logging
from warnings import warn

from swyft.store import MemoryStore
from swyft.inference import DefaultHead, DefaultTail, RatioCollection, JoinedRatioCollection
from swyft.ip3 import Dataset
from swyft.marginals import PosteriorCollection

logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class Posteriors:
    def __init__(self, dataset, simhook = None, device = 'cpu'):
        # Store relevant information about dataset
        self._bounded_prior = dataset.bounded_prior
        self._indices = dataset.indices
        self._N = len(dataset)
        self._ratios = []

        # Temporary
        self._dataset = dataset
        self._device = device

    def infer(
        self,
        partitions, 
        train_args: dict = {},
        head=DefaultHead,
        tail=DefaultTail,
        head_args: dict = {},
        tail_args: dict = {},
        max_rounds: int = 10,
        keep_history=False,
    ):
        """Perform 1-dim marginal focus fits.

        Args:
            train_args (dict): Training keyword arguments.
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
            max_rounds (int): Maximum number of rounds per invokation of `run`, default 10.
        """
        ntrain = self._N
        bp = self._bounded_prior.bound

        re = self._train(
            bp,
            partitions,
            head=head,
            tail=tail,
            head_args=head_args,
            tail_args=tail_args,
            train_args=train_args,
            N=ntrain,
        )
        self._ratios.append(re)

    def sample(self, N, obs):
        post = PosteriorCollection(self.ratios, self._bounded_prior)
        samples = post.sample(N, obs)
        return samples

    @property
    def bound(self):
        return self._bounded_prior.bound

    @property
    def ptrans(self):
        return self._bounded_prior.ptrans

    @property
    def ratios(self):
        return JoinedRatioCollection(self._ratios[::-1])

    def _train(
        self,
        prior,
        param_list,
        N,
        train_args,
        head,
        tail,
        head_args,
        tail_args,
    ):
        if param_list is None:
            param_list = prior.params()

        re = RatioCollection(
            param_list,
            device=self._device,
            head=head,
            tail=tail,
            tail_args=tail_args,
            head_args=head_args,
        )
        re.train(self._dataset, **train_args)

        return re

    def state_dict(self):
        raise NotImplementedError

    @classmethod
    def from_state_dict(cls, state_dict):
        raise NotImplementedError
