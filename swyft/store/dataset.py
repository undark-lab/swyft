import logging

import numpy as np
import torch
from torch.utils.data import Dataset as torch_Dataset

import swyft

log = logging.getLogger(__name__)


class Dataset(torch_Dataset):
    """Dataset for access to swyft.Store.

    Args:
        N (int): Number of samples.
        prior (swyft.Prior): Parameter prior.
        store (swyft.Store): Store reference.
        simhook (Callable): Posthook for simulations. Applied on-the-fly to each point.
        simkeys (list of strings): List of simulation keys that should be exposed 
                                    (None means that all store sims are exposed).
        add (bool): If necessary, automatically add new points to the store.

    .. note::
        swyft.Dataset is essentially a list of indices that point to
        corresponding entries in the swyft.Store.  It is a daugther class of
        torch.utils.data.Dataset, and can be used directly for training.  Due
        to the statistical nature of the Store, the returned number of samples
        is effectively drawn from a Poisson distribution with mean N.  
    """

    def __init__(self, N, prior, store, bound=None, simhook=None, simkeys = None, add = False):
        super().__init__()

        # Initialization
        self._trunc_prior = swyft.TruncatedPrior(prior, bound)
        self._indices = store.sample(N, prior, bound=bound, add = add)

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
    def prior(self):
        """Return prior of dataset (swyft.Prior)."""
        return self._trunc_prior.prior

    @property
    def bound(self):
        """Return bound of truncated prior of dataset (swyft.Bound)."""
        return self._trunc_prior.bound

    def _tensorfy(self, x):
        return {k: torch.tensor(v).float() for k, v in x.items()}

    @property
    def indices(self):
        """Return indices of the dataset that indicate positions in the store."""
        return self._indices

#    def _no_store(self):
#        if self._store is None:
#            print("WARNING: No store defined.")
#            return True
#        else:
#            return False

    def simulate(self, batch_size=None, wait_for_results=True):
        """Trigger simulations for points in the dataset.

        Args:
            batch_size (int): Number of batched simulations.
            wait_for_results (bool): What for simulations to complete before returning.
        """
#        if self._no_store():
#            return
        self._store.simulate(
            self.indices, batch_size=batch_size, wait_for_results=wait_for_results
        )

#    def set_store(self, store):
#        self._store = store

    @property
    def requires_sim(self):
        """Check if simulations are required for points in the dataset."""
#        if self._no_store():
#            return
        return self._store.requires_sim(self.indices)

    @property
    def v(self):
        """Return all parameters as npoints x zdim array."""
#        if self._no_store():
#            return
        return np.array([self._store.v[i] for i in self._indices])

    @property
    def pnames(self):
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
        u = self._trunc_prior.prior.u(v.reshape(1,-1)).flatten()

        return (self._tensorfy(x), torch.tensor(u).float(), torch.tensor(v).float())

    def state_dict(self):
        return dict(
            indices=self._indices,
            trunc_prior=self._trunc_prior.state_dict(),
            simhook=bool(self._simhook),
            simkeys=self._simkeys
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
        obj._simkeys = state_dict['simkeys']
        if state_dict["simhook"] and not simhook:
            log.warning(
                "A simhook was specified when the dataset was saved, but is missing now."
            )
        if not state_dict["simhook"] and simhook:
            log.warning(
                "A simhook was specified, but no simhook was specified when the Dataset was saved."
            )
        return obj

    def save(self, filename):
        """Save dataset (including indices).

        Args:
            filename (str): Output filename
        """
        torch.save(self.state_dict(), filename)

    @classmethod
    def load(cls, filename, store, simhook=None):
        """Load dataset.

        Args:
            filename (str)
            store (swyft.Store): Corresponding datastore.
            simhook (callable): Simulation hook.

        .. warning::
            Make sure that the store is the same that was originally used for
            creating the dataset.
        """
        sd = torch.load(filename)
        return cls.from_state_dict(sd, store, simhook=simhook)


# TODO: Needs to be updated
class ExactDataset(Dataset):
    """Dataset with exactly a certain number of simulations."""

    def __init__(self, N, prior, store, simhook=None, oversample_factor: float = 1.2):
        """Initialize Dataset.

        Args:
            N (int): Number of samples.
            prior (swyft.Prior): Parameter prior.
            store (swyft.Store): Store reference.
            simhook (Callable): Posthook for simulations. Applied on-the-fly to each point.
            oversample_factor (float): how many extra samples to draw (to make sure we can subsample to get an exact length)
        """
        super().__init__(int(oversample_factor * N), prior, store, simhook=simhook)
        while len(self) < N:
            indices = store.sample(int(oversample_factor * N), prior)
            self._indices = indices
        self._indices = self._indices[:N]

    @classmethod
    def from_parent_dataset(
        cls, N, parent_dataset: Dataset, oversample_factor: float = 1.2
    ):
        """Initialize Dataset from a parent. Story is copied and simulations are help in in oversampled memory store.

        Args:
            N (int): number of samples.
            parent_dataset (Dataset): where to draw the prior and copy the store from.
            oversample_factor (float): how many extra samples to draw (to make sure we can subsample to get an exact length)
        """
        prior = parent_dataset.prior
        store = parent_dataset._store.copy()
        simhook = parent_dataset._simhook
        dataset = cls(int(oversample_factor * N), prior, store, simhook)

        counter = 0
        while len(dataset) < N:
            dataset._indices = dataset._store.sample(int(oversample_factor * N), prior)
            counter += 1

            if counter > 10:
                raise RuntimeError()
        dataset._indices = dataset._indices[:N]
        return dataset
