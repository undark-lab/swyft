# pylint: disable=no-member, not-callable
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import zarr
import numcodecs
from tqdm import tqdm

from .utils import is_empty
from .types import Shape, Union, Sequence, PathType


class Cache(ABC):
    _filesystem = (
        "metadata",
        "metadata/intensity",
        "metadata/requires_simulation",
        "samples",
        "samples/x",
        "samples/z",
    )

    @abstractmethod
    def __init__(
        self,
        zdim: int,
        xshape: Shape,
        store: Union[zarr.MemoryStore, zarr.DirectoryStore],
    ):
        """Initialize Cache content dimensions.

        Args:
            zdim (int): Number of z dimensions
            xshape (tuple): Shape of x array
            store (zarr.MemoryStore, zarr.DirectoryStore)
        """
        self._zdim = None
        self._xshape = None
        self.store = store
        self.root = zarr.group(store=self.store)

        if all(key in self.root.keys() for key in ["samples", "metadata"]):
            print("Loading existing cache.")
            self._update()
        elif len(self.root.keys()) == 0:
            print("Creating new cache.")
            self.x = self.root.zeros(
                self._filesystem[4],  # samples/x
                shape=(0, *xshape),
                chunks=(1, *xshape),
                dtype="f4",
            )
            self.z = self.root.zeros(
                self._filesystem[5],  # samples/z
                shape=(0, zdim),
                chunks=(10000, zdim),
                dtype="f4",
            )
            self.m = self.root.zeros(
                self._filesystem[2],  # metadata/requires_simulation
                shape=(0, 1),
                chunks=(10000, 1),
                dtype="bool",
            )
            self.u = self.root.create(
                self._filesystem[1],  # metadata/intensity
                shape=(0,),
                dtype=object,
                object_codec=numcodecs.Pickle(),
            )
        else:
            raise KeyError(
                "The zarr storage is corrupted. It should either be empty or only have the keys ['samples', 'metadata']."
            )

        assert (
            zdim == self.zdim
        ), f"Your given zdim, {zdim}, was not equal to the one defined in zarr {self.zdim}."
        assert (
            xshape == self.xshape
        ), f"Your given xshape, {xshape}, was not equal to the one defined in zarr {self.xshape}."

    @staticmethod
    def _extract_xshape_from_zarr_group(group):
        return group["samples/x"].shape[1:]

    @staticmethod
    def _extract_zdim_from_zarr_group(group):
        return group["samples/z"].shape[1]

    @property
    def xshape(self):
        if self._xshape is None:
            self._xshape = self._extract_xshape_from_zarr_group(self.root)
        return self._xshape

    @property
    def zdim(self):
        if self._zdim is None:
            self._zdim = self._extract_zdim_from_zarr_group(self.root)
        return self._zdim

    def _update(self):
        # This could be removed with a property for each attribute which only loads from disk if something has changed. TODO
        self.x = self.root["samples/x"]
        self.z = self.root["samples/z"]
        self.m = self.root["metadata/requires_simulation"]
        self.u = self.root["metadata/intensity"]

    def __len__(self):
        """Returns number of samples in the cache."""
        self._update()
        return len(self.z)

    def __getitem__(self, i):
        self._update()
        return self.x[i], self.z[i]

    def _append_z(self, z):
        """Append z to cache content and new slots for x."""
        self._update()

        # Add simulation slots
        xshape = list(self.x.shape)
        xshape[0] += len(z)
        self.x.resize(*xshape)

        # Add z samples
        self.z.append(z)

        # Register as missing
        m = np.ones((len(z), 1), dtype="bool")
        self.m.append(m)

    def intensity(self, zlist):
        """Evaluate Cache intensity function.

        Args:
            z (array-like): list of parameter values.
        """
        self._update()

        if len(self.u) == 0:
            return np.zeros(len(zlist))
        else:
            return np.array([self.u[i](zlist) for i in range(len(self.u))]).max(axis=0)

    def grow(self, p):
        """Grow number of samples in cache."""
        # Proposed new samples z from p
        z_prop = p.sample()

        # Rejection sampling from proposal list
        accepted = []
        ds_intensities = self.intensity(z_prop)
        target_intensities = p(z_prop)
        for _, Ids, It in zip(z_prop, ds_intensities, target_intensities):
            rej_prob = np.minimum(1, Ids / It)
            w = np.random.rand()
            accepted.append(rej_prob < w)
        z_accepted = z_prop[accepted, :]

        # Add new entries to cache
        if len(z_accepted) > 0:
            self._append_z(z_accepted)
            print("Adding %i new samples. Run simulator!" % len(z_accepted))
        else:
            print("No new simulator runs required.")

        # save intensity function. We collect them all to find their maximum.
        self.u.resize(len(self.u) + 1)
        self.u[-1] = p

    def sample(self, p):
        """Sample from Cache.

        Args:
            p (intensity function): Target intensity function.
        """
        self._update()

        self.grow(p)

        accepted = []
        zlist = self.z[:]
        I_ds = self.intensity(zlist)
        I_target = p(zlist)
        for i, z in enumerate(zlist):
            accept_prob = I_target[i] / I_ds[i]
            assert accept_prob <= 1.0, (
                f"{accept_prob} > 1, but we expected the ratio of target intensity function to the cache <= 1. "
                "There may not be enough samples in the cache "
                "or a constrained intensity function was not accounted for."
            )
            w = np.random.rand()
            if accept_prob > w:
                accepted.append(i)
        return accepted

    def _require_sim_idx(self):
        indices = []
        m = self.m[:]
        for i in range(len(self.z)):
            if m[i]:
                indices.append(i)
        return indices

    def requires_sim(self):
        self._update()

        return len(self._require_sim_idx()) > 0

    def _add_sim(self, i, x):
        self.x[i] = x
        self.m[i] = False

    def simulate(self, simulator):
        """Run simulator sequentially for missing points.

        Args:
            simulator (callable): Simulator
        """
        self._update()

        idx = self._require_sim_idx()
        if len(idx) == 0:
            print("No simulations required.")
            return
        for i in tqdm(idx, desc="Simulate"):
            z = self.z[i]
            x = simulator(z)
            self._add_sim(i, x)


class DirectoryCache(Cache):
    def __init__(self, zdim: int, xshape: Shape, path: PathType):
        """Instantiate an iP3 cache stored in a directory.

        Args:
            zdim (int): Number of z dimensions
            xshape (tuple): Shape of x array
            path: path to storage directory
        """
        self.store = zarr.DirectoryStore(path)
        super().__init__(zdim=zdim, xshape=xshape, store=self.store)

    @classmethod
    def load(cls, path: PathType):
        """Load existing DirectoryStore."""
        store = zarr.DirectoryStore(path)
        group = zarr.group(store=store)
        xshape = cls._extract_xshape_from_zarr_group(group)
        zdim = cls._extract_zdim_from_zarr_group(group)
        return DirectoryCache(zdim=zdim, xshape=xshape, path=path)


class MemoryCache(Cache):
    def __init__(self, zdim: int, xshape: Shape, store=None):
        """Instantiate an iP3 cache stored in the memory.

        Args:
            zdim (int): Number of z dimensions
            xshape (tuple): Shape of x array
            store (zarr.MemoryStore, zarr.DirectoryStore): optional, used in loading.
        """
        if store is None:
            self.store = zarr.MemoryStore()
        else:
            self.store = store
        super().__init__(zdim=zdim, xshape=xshape, store=self.store)

    def save(self, path: PathType):
        """Copy the current state of the MemoryCache to a directory."""
        path = Path(path)
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"{path} should be a directory")
        elif path.exists() and not is_empty(path):
            raise FileExistsError(f"{path} is not empty")
        else:
            path.mkdir(parents=True, exist_ok=True)
            store = zarr.DirectoryStore(path)
            zarr.convenience.copy_store(source=self.store, dest=store)
            return None

    @classmethod
    def load(cls, path: PathType):
        """Copy existing DirectoryStore state into a MemoryCache object."""
        memory_store = zarr.MemoryStore()
        directory_store = zarr.DirectoryStore(path)
        zarr.convenience.copy_store(source=directory_store, dest=memory_store)

        group = zarr.group(store=memory_store)
        xshape = cls._extract_xshape_from_zarr_group(group)
        zdim = cls._extract_zdim_from_zarr_group(group)
        return MemoryCache(zdim=zdim, xshape=xshape, store=memory_store)


if __name__ == "__main__":
    pass
