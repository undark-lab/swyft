from typing import Callable, Hashable, Optional, Sequence, Tuple
from warnings import warn

import numpy as np
import torch

import swyft
from swyft.saveable import StateDictSaveable
from swyft.types import Array, ObsType, ParameterNamesType, PathType
from swyft.utils.array import array_to_tensor


class SimpleDataset(torch.utils.data.Dataset):
    """Dataset which merely keeps track of observation and parameter data in one place"""

    def __init__(self, observations: ObsType, us: Array, vs: Array) -> None:
        """
        Args:
            observations: dictionary of batched obserations
            us: array or tensor of unit cube parameters
            vs: array or tensor of natural parameters
        """
        super().__init__()
        b = us.shape[0]
        assert (
            vs.shape[0] == b
        ), "the us and vs arrays do not have the same batch dimension"
        assert all(
            [b == x.shape[0] for x in observations.values()]
        ), "the observation values do not have the same batch dimension as us and vs"
        self.observations = observations
        self.us = us
        self.vs = vs
        self._len = b

    def __getitem__(self, idx) -> Tuple[ObsType, torch.Tensor, torch.Tensor]:
        return (
            {
                key: array_to_tensor(val[idx, ...])
                for key, val in self.observations.items()
            },
            array_to_tensor(self.us[idx, ...]),
            array_to_tensor(self.vs[idx, ...]),
        )

    def __len__(self) -> int:
        return self._len


class Dataset(torch.utils.data.Dataset, StateDictSaveable):
    """Dataset for access to swyft.Store."""

    def __init__(
        self,
        N: int,
        prior: swyft.Prior,
        store: "swyft.store.store.Store",
        bound: Optional[swyft.Bound] = None,
        simhook: Optional[Callable[..., ObsType]] = None,
        simkeys: Optional[Sequence[Hashable]] = None,
    ) -> None:
        """
        Args:
            N: Number of samples.
            prior: Parameter prior.
            store: Store reference.
            simhook: Applied on-the-fly to each sample. simhook(x, v)
            simkeys: List of simulation keys that should be exposed (None means that all store sims are exposed)

        .. note::
            swyft.Dataset is essentially a list of indices that point to
            corresponding entries in the swyft.Store.  It is a daughter class of
            torch.utils.data.Dataset, and can be used directly for training.  Due
            to the statistical nature of the Store, the returned number of samples
            is effectively drawn from a Poisson distribution with mean N.
        """
        super().__init__()
        self._prior_truncator = swyft.PriorTruncator(prior, bound)
        self.indices = store.sample(N, prior, bound=bound)
        self._store = store
        self._simhook = simhook
        self._simkeys = simkeys if simkeys else list(self._store.sims)

        if self.requires_sim:
            warn(
                "The store requires simulation for some of the indices in this Dataset."
            )

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.indices)

    @property
    def prior(self) -> swyft.Prior:
        """Return prior of dataset."""
        return self._prior_truncator.prior

    @property
    def bound(self) -> swyft.Bound:
        """Return bound of truncated prior of dataset (swyft.Bound)."""
        return self._prior_truncator.bound

    def simulate(
        self, batch_size: Optional[int] = None, wait_for_results: bool = True
    ) -> None:
        """Trigger simulations for points in the dataset.

        Args:
            batch_size: Number of batched simulations.
            wait_for_results: What for simulations to complete before returning.
        """
        self._store.simulate(
            self.indices, batch_size=batch_size, wait_for_results=wait_for_results
        )

    @property
    def requires_sim(self) -> bool:
        """Check if simulations are required for points in the dataset."""
        return self._store.requires_sim(self.indices)

    @property
    def v(self) -> np.ndarray:
        """Return all parameters as (n_points, n_parameters) array."""
        return np.array([self._store.v[i] for i in self.indices])

    @property
    def parameter_names(self) -> ParameterNamesType:
        """Return parameter names (inherited from store and simulator)."""
        return self._store.parameter_names

    def __getitem__(self, idx) -> Tuple[ObsType, torch.Tensor, torch.Tensor]:
        """Return datastore entry."""
        i = self.indices[idx]
        x = {k: self._store.sims[k][i] for k in self._simkeys}
        v = self._store.v[i]
        if self._simhook is not None:
            x = self._simhook(x, v)
        u = self.prior.cdf(v.reshape(1, -1)).flatten()

        return (
            {key: array_to_tensor(val) for key, val in x.items()},
            array_to_tensor(u),
            array_to_tensor(v),
        )

    def state_dict(self) -> dict:
        return dict(
            indices=self.indices,
            prior_truncator=self._prior_truncator.state_dict(),
            simhook=bool(self._simhook),
            simkeys=self._simkeys,
        )

    @classmethod
    def from_state_dict(cls, state_dict, store, simhook=None):
        obj = cls.__new__(cls)
        obj._prior_truncator = swyft.PriorTruncator.from_state_dict(
            state_dict["prior_truncator"]
        )
        obj.indices = state_dict["indices"]

        obj._store = store
        obj._simhook = simhook
        obj._simkeys = state_dict["simkeys"]
        if simhook is None and state_dict["simhook"] is True:
            warn(
                "A simhook was specified when the dataset was saved, but is missing now."
            )
        elif simhook is not None and state_dict["simhook"] is False:
            warn(
                "A simhook was specified, but no simhook was specified when the Dataset was saved."
            )
        return obj

    def save(self, filename: PathType) -> None:
        """
        .. note::
            The store and the simhook are not saved. They must be loaded independently by the user.
        """
        sd = self.state_dict()
        torch.save(sd, filename)

    @classmethod
    def load(
        cls,
        filename: PathType,
        store: "swyft.Store",
        simhook: Callable[..., ObsType] = None,
    ):
        """Load dataset.

        Args:
            filename
            store: Corresponding datastore.
            simhook: Simulation hook.

        .. warning::
            Make sure that the store is the same that was originally used for
            creating the dataset.
        """
        sd = torch.load(filename)
        return cls.from_state_dict(sd, store, simhook=simhook)
