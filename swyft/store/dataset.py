import logging
from typing import Callable, Hashable, Optional, Sequence
from warnings import warn

import numpy as np
import torch

import swyft
from swyft.types import ObsType, ParameterNamesType, PathType
from swyft.utils.array import array_to_tensor

log = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    """Dataset for access to swyft.Store.

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

    def __init__(
        self,
        N: int,
        prior: swyft.Prior,
        store: "swyft.store.store.Store",
        bound: Optional[swyft.Bound] = None,
        simhook: Optional[Callable[..., ObsType]] = None,
        simkeys: Optional[Sequence[Hashable]] = None,
    ) -> None:
        super().__init__()
        self._trunc_prior = swyft.TruncatedPrior(
            prior, bound
        )  # TODO why do we need this?
        self.indices = store.sample(N, prior, bound=bound)
        self._store = store
        self._simhook = simhook
        self._simkeys = simkeys if simkeys else list(self._store.sims)

        if self.requires_sim:
            warn(
                "The store requires simulation for some of the indices in this Dataset."
            )

    def __len__(self):
        """Return length of dataset."""
        return len(self.indices)

    @property
    def prior(self) -> swyft.Prior:
        """Return prior of dataset."""
        return self._trunc_prior.prior

    @property
    def bound(self) -> swyft.Bound:
        """Return bound of truncated prior of dataset (swyft.Bound)."""
        return self._trunc_prior.bound

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

    def __getitem__(self, idx):
        """Return datastore entry."""
        i = self.indices[idx]
        x = {k: self._store.sims[k][i] for k in self._simkeys}
        v = self._store.v[i]
        if self._simhook is not None:
            x = self._simhook(x, v)
        u = self._trunc_prior.prior.u(v.reshape(1, -1)).flatten()

        return (
            {k: array_to_tensor(v) for k, v in x.items()},
            array_to_tensor(u),
            array_to_tensor(v),
        )

    def state_dict(self) -> dict:
        return dict(
            indices=self.indices,
            trunc_prior=self._trunc_prior.state_dict(),
            simhook=bool(self._simhook),
            simkeys=self._simkeys,
        )

    @classmethod
    def from_state_dict(cls, state_dict, store, simhook=None):
        obj = cls.__new__(cls)
        obj._trunc_prior = swyft.TruncatedPrior.from_state_dict(
            state_dict["trunc_prior"]
        )
        obj.indices = state_dict["indices"]

        obj._store = store
        obj._simhook = simhook
        obj._simkeys = state_dict["simkeys"]
        if simhook is None and state_dict["simhook"] is not None:
            warn(
                "A simhook was specified when the dataset was saved, but is missing now."
            )
        elif simhook is not None and state_dict["simhook"] is None:
            warn(
                "A simhook was specified, but no simhook was specified when the Dataset was saved."
            )
        return obj

    def save(self, filename: PathType) -> None:
        """Save dataset (including indices).

        Args:
            filename: Output filename
        """
        torch.save(self.state_dict(), filename)

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
