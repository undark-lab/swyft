from typing import Callable, List

import numpy as np
from torch.utils.data import Dataset as torch_Dataset
import torch


class Dataset(torch_Dataset):
    def __init__(self, N, bounded_prior, store, simhook = None):
        # Initialization
        store.grow(N, bounded_prior)
        indices = store.sample(N, bounded_prior)

        self._bounded_prior = bounded_prior
        self._indices = indices

        self._store = store
        self._simhook = simhook

    def __len__(self):
        return len(self._indices)

    @property
    def bounded_prior(self):
        return self._bounded_prior

    def _tensorfy(self, x):
        return {k: torch.tensor(v).float() for k, v in x.items()}

    @property
    def indices(self):
        return self._indices

    def simulate(self):
        self._store.simulate()

    def __getitem__(self, idx):
        i = self._indices[idx]
        x_keys = list(self._store.x)
        z_keys = list(self._store.z)
        x = {k: self._store.x[k][i] for k in x_keys}
        z = self._store.z[i]
        u = self._bounded_prior.ptrans.u(z.reshape(1, -1)).flatten()

        if self._simhook is not None:
            x = self._simhook(x, z)

        return dict(obs=self._tensorfy(x), par=torch.tensor(u).float())


#class Points:
#    """Points references (observation, parameter) pairs drawn from an inhomogenous Poisson Point Proccess (iP3) Store.
#    Points implements this via a list of indices corresponding to data contained in a store which is provided at initialization.
#    """
#
#    _save_attrs = ["indices"]
#
#    def __init__(
#        self, 
#        store: "swyft.store.Store",
#        ptrans,
#        indices: List[int] = None,
#        noisehook: Callable = None
#    ):  # noqa: F821
#        """Create a points dataset
#
#        Args:
#            store (Store): iP3 store for zarr storage
#            intensity (Intensity): inhomogenous Poisson Point Proccess intensity function on parameters
#            noisehook (function): (optional) maps from (x, z) to x with noise
#        """
#        if store.any_failed:
#            raise RuntimeError(
#                "The store has parameters which failed to return a simulation. Try resampling them."
#            )
#        elif store.requires_sim:
#            raise RuntimeError(
#                "The store has parameters without a corresponding observation. Try running the simulator."
#            )
#        if indices is None:
#            indices = range(len(store))
#        assert (
#            len(indices) != 0
#        ), "You passed indices with length zero. That implies no points."
#
#        self.store = store
#        self.noisehook = noisehook
#        self.indices = np.array(indices)
#        self.ptrans = ptrans
#
#    def __len__(self):
#        return len(self.indices)
#
#    def params(self):
#        return self.store.params
#
#    def get_range(self, indices):
#        N = len(indices)
#        obs_comb = {k: np.empty((N,) + v.shape) for k, v in self[0]["obs"].items()}
#        par_comb = {k: np.empty((N,) + v.shape) for k, v in self[0]["par"].items()}
#
#        for i in indices:
#            p = self[i]
#            for k, v in p["obs"].items():
#                obs_comb[k][i] = v
#            for k, v in p["par"].items():
#                par_comb[k][i] = v
#
#        return dict(obs=obs_comb, par=par_comb)
#
#    def __getitem__(self, idx):
#        i = self.indices[idx]
#        x_keys = list(self.store.x)
#        z_keys = list(self.store.z)
#        x = {k: self.store.x[k][i] for k in x_keys}
#        z = self.store.z[i]
#        u = self.ptrans.u(z.reshape(1, -1)).flatten()
#
#        if self.noisehook is not None:
#            x = self.noisehook(x, z)
#
#        return dict(obs=x, par=u)
