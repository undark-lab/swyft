import logging
from typing import Callable, Dict, Hashable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset as torch_Dataset

import swyft
from swyft.types import Array, ObsType, PathType, PNamesType
from swyft.utils.array import array_to_tensor

log = logging.getLogger(__name__)


class Dataset(torch_Dataset):
    """Dataset for access to swyft.Store.

    Args:
        N: Number of samples.
        prior: Parameter prior.
        store: Store reference.
        simhook : Posthook for simulations. Applied on-the-fly to each point.
        simkeys: List of simulation keys that should be exposed
                    (None means that all store sims are exposed).

    .. note::
        swyft.Dataset is essentially a list of indices that point to
        corresponding entries in the swyft.Store.  It is a daugther class of
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

        # Initialization
        self._trunc_prior = swyft.TruncatedPrior(prior, bound)
        self._indices = store.sample(N, prior, bound=bound)
        if len(self._indices) == 0:
            raise RuntimeError("Not enough simulations in store")

        self._store = store
        self._simhook = simhook
        self._simkeys = simkeys if simkeys else list(self._store.sims)
        self._pnames = self._store.pnames

        if self.requires_sim:
            print("WARNING: Some points require simulation.")

    def __len__(self):
        """Return length of dataset."""
        return len(self._indices)

    @property
    def prior(self) -> swyft.Prior:
        """Return prior of dataset."""
        return self._trunc_prior.prior

    @property
    def bound(self) -> swyft.Bound:
        """Return bound of truncated prior of dataset (swyft.Bound)."""
        return self._trunc_prior.bound

    def _tensorfy(self, x: Dict[Hashable, Array]) -> Dict[Hashable, torch.Tensor]:
        return {k: array_to_tensor(v) for k, v in x.items()}

    @property
    def indices(self) -> np.ndarray:
        """Return indices of the dataset that indicate positions in the store."""
        return self._indices

    def simulate(
        self, batch_size: Optional[int] = None, wait_for_results: bool = True
    ) -> None:
        """Trigger simulations for points in the dataset.

        Args:
            batch_size: Number of batched simulations.
            wait_for_results: What for simulations to complete before returning.
        """
        #        if self._no_store():
        #            return
        self._store.simulate(
            self.indices, batch_size=batch_size, wait_for_results=wait_for_results
        )

    @property
    def requires_sim(self) -> bool:
        """Check if simulations are required for points in the dataset."""
        #        if self._no_store():
        #            return
        return self._store.requires_sim(self.indices)

    @property
    def v(self) -> np.ndarray:
        """Return all parameters as npoints x zdim array."""
        #        if self._no_store():
        #            return
        return np.array([self._store.v[i] for i in self._indices])

    @property
    def pnames(self) -> PNamesType:
        """Return parameter names (inherited from store and simulator)."""
        return self._pnames

    def __getitem__(self, idx):
        """Return datastore entry."""
        #        if self._no_store():
        #            return
        i = self._indices[idx]
        x_keys = self._simkeys
        x = {k: self._store.sims[k][i] for k in x_keys}
        v = self._store.v[i]
        if self._simhook is not None:
            x = self._simhook(x, v)
        u = self._trunc_prior.prior.u(v.reshape(1, -1)).flatten()

        return (
            self._tensorfy(x),
            array_to_tensor(u),
            array_to_tensor(v),
        )

    def state_dict(self) -> dict:
        return dict(
            indices=self._indices,
            trunc_prior=self._trunc_prior.state_dict(),
            simhook=bool(self._simhook),
            simkeys=self._simkeys,
            pnames=self._pnames,
        )

    @classmethod
    def from_state_dict(cls, state_dict, store, simhook=None):
        obj = Dataset.__new__(Dataset)
        obj._trunc_prior = swyft.TruncatedPrior.from_state_dict(
            state_dict["trunc_prior"]
        )
        obj._indices = state_dict["indices"]

        obj._store = store
        obj._simhook = simhook
        obj._simkeys = state_dict["simkeys"]
        obj._pnames = state_dict["pnames"]
        if state_dict["simhook"] and not simhook:
            log.warning(
                "A simhook was specified when the dataset was saved, but is missing now."
            )
        if not state_dict["simhook"] and simhook:
            log.warning(
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
