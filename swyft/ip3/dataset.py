from typing import Callable, List

import logging

import numpy as np
from torch.utils.data import Dataset as torch_Dataset
from swyft.marginals.prior import Prior
import torch


class Dataset(torch_Dataset):
    def __init__(self, N, prior, store, simhook = None):
        super().__init__()

        # Initialization
        indices = store.sample(N, prior)

        self._prior = prior
        self._indices = indices

        self._store = store
        self._simhook = simhook


    def __len__(self):
        return len(self._indices)

    @property
    def prior(self):
        return self._prior

    def _tensorfy(self, x):
        return {k: torch.tensor(v).float() for k, v in x.items()}

    @property
    def indices(self):
        return self._indices

    def simulate(self):
        self._store.simulate(self.indices)

    @property
    def requires_sim(self):
        return self._store.requires_sim(self.indices)

    @property
    def z(self):
        return np.array([self._store.z[i] for i in self._indices])

    def __getitem__(self, idx):
        i = self._indices[idx]
        x_keys = list(self._store.x)
        x = {k: self._store.x[k][i] for k in x_keys}
        z = self._store.z[i]
        u = self._prior.ptrans.u(z.reshape(1, -1)).flatten()

        if self._simhook is not None:
            x = self._simhook(x, z)

        return (self._tensorfy(x), torch.tensor(u).float())

    def state_dict(self):
        return dict(indices = self._indices,
                prior = self._prior.state_dict(),
                simhook = bool(self._simhook)
                )

    @classmethod
    def from_state_dict(cls, state_dict, store = None, simhook = None):
        obj = Dataset.__new__(Dataset)
        obj._prior = Prior.from_state_dict(state_dict['prior'])
        obj._indices = state_dict['indices']

        obj._store = store
        if store is None:
            logging.warning("No store specified!")
        obj._simhook = simhook
        if state_dict['simhook'] and not simhook:
            logging.warning("A simhook was specified when the dataset was saved, but is missing now.")
        if not state_dict['simhook'] and simhook:
            logging.warning("A simhook was specified, but no simhook was specified when the Dataset was saved.")
        return obj

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    @classmethod
    def load(cls, filename, store = None, simhook = None):
        sd = torch.load(filename)
        return cls.from_state_dict(sd, store = store, simhook = simhook)



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
