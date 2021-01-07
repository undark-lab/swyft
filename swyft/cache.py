# pylint: disable=no-member, not-callable
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import zarr
import numcodecs
from tqdm import tqdm

from .utils import is_empty
from .types import Shape, Union, Sequence, PathType, Callable, Array

from .intensity import IntensityNew

class MissingSimulationError(Exception):
    pass

class LowIntensityError(Exception):
    pass

class Cache(ABC):
    """Abstract base class for various caches."""

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
        param_names,
        obs_shapes: Shape,
        store: Union[zarr.MemoryStore, zarr.DirectoryStore],
    ):
        """Initialize Cache content dimensions.

        Args:
            param_names (list of strings): List of paramater names 
            obs_shapes (dict): Map of obs names to shapes
            store: zarr storage.
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
            self._setup_new_cache(param_names, obs_shapes, self.root)
        else:
            raise KeyError(
                "The zarr storage is corrupted. It should either be empty or only have the keys ['samples', 'metadata']."
            )

        #assert (
        #    zdim == self.zdim
        #), f"Your given zdim, {zdim}, was not equal to the one defined in zarr {self.zdim}."
        #assert (
        #    xshape == self.xshape
        #), f"Your given xshape, {xshape}, was not equal to the one defined in zarr {self.xshape}."

    def _setup_new_cache(self, param_names, obs_shapes, root):
        # Add parameter names to store
        z = root.create_group(self._filesystem[5])
        for name in param_names:
            z.zeros(name, shape=(0,), chunks=(100000,), dtype="f4")
            # FIX: Too mall chunks lead to problems with appending

        # Adding observational shapes to store
        x = root.create_group(self._filesystem[4])
        for name, shape in obs_shapes.items():
            x.zeros(name, shape=(0, *shape), chunks=(1, *shape), dtype="f4")

        # Requires simulation flag
        m = root.zeros(
            self._filesystem[2],  # metadata/requires_simulation
            shape=(0, 1),
            chunks=(100000, 1),
            dtype="bool",
        )

        # Intensity object
        u = root.create(
            self._filesystem[1],  # metadata/intensity
            shape=(0,),
            dtype=object,
            object_codec=numcodecs.Pickle(),
        )

        return dict(u=u, m=m, x=x, z=z)

    @staticmethod
    def _extract_xshape_from_zarr_group(group):
        return group["samples/x"].shape[1:]

    @staticmethod
    def _extract_zdim_from_zarr_group(group):
        return group["samples/z"].shape[1]

    @property
    def param_names(self):
        return list(self.z)

    @property
    def xshape(self) -> Shape:
        """Shape of observations."""
        if self._xshape is None:
            self._xshape = self._extract_xshape_from_zarr_group(self.root)
        return self._xshape

    @property
    def zdim(self) -> int:
        """Dimension of parameters."""
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
        # Return len of first entry
        param_names = list(self.z)
        return len(self.z[param_names[0]])

    def __getitem__(self, i):
        self._update()

        result_x = {}
        for key, value in self.x.items():
            result_x[key] = value[i]

        result_z = {}
        for key, value in self.z.items():
            result_z[key] = value[i]

        return dict(x=result_x, z=result_z)

    def _append_z(self, z):
        """Append z to cache content and new slots for x."""
        self._update()

        # Length of first element
        n = len(z[list(z)[0]])

        # Add slots for x
        for key, value in self.x.items():
            shape = list(value.shape)
            shape[0] += n
            value.resize(*shape)

        # Add z samples
        for key, value in self.z.items():
            print(z[key].shape)
            value.append(z[key])

        # Register as missing
        m = np.ones((n, 1), dtype="bool")
        self.m.append(m)

    def intensity(self, z: Array) -> np.ndarray:
        """Evaluate the cache's intensity function.

        Args:
            z: list of parameter values.
        """
        self._update()

        if len(self.u) == 0:
            d = len(z[list(z)[0]])
            return np.zeros(d)
        else:
            return np.array([self.u[i](z) for i in range(len(self.u))]).max(axis=0)

    def grow(self, prior: "swyft.intensity.Intensity", N):
        """Given an intensity function, add parameter samples to the cache.

        Args:
            intensity: target parameter intensity function
        """
        intensity = IntensityNew(prior, N)

        # Proposed new samples z from p
        z_prop = intensity.sample()

        # Rejection sampling from proposal list
        accepted = []
        ds_intensities = self.intensity(z_prop)
        target_intensities = intensity(z_prop)
        for Ids, It in zip(ds_intensities, target_intensities):
            rej_prob = np.minimum(1, Ids / It)
            w = np.random.rand()
            accepted.append(rej_prob < w)
        z_accepted = {k: z[accepted, ...] for k, z in z_prop.items()}

        # Add new entries to cache
        if sum(accepted) > 0:
            self._append_z(z_accepted)
            print("Adding %i new samples. Run simulator!" % sum(accepted))
        else:
            print("No new simulator runs required.")

        # save intensity function. We collect them all to find their maximum.
        self.u.resize(len(self.u) + 1)
        self.u[-1] = intensity

    def sample(self, prior: "swyft.intensity.Intensity", N):
        """Sample from Cache.

        Args:
            intensity: target parameter intensity function
        """
        intensity = IntensityNew(prior, N)

        self._update()

        #self.grow(prior, N)

        accepted = []
        zlist = {k: self.z[k][:] for k in (self.z)}

        I_ds = self.intensity(zlist)
        I_target = intensity(zlist)
        for i in range(len(self)):
            accept_prob = I_target[i] / I_ds[i]
            if accept_prob > 1.0:
                raise LowIntensityError(
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
        for i in range(len(self)):
            if m[i]:
                indices.append(i)
        return indices

    def requires_sim(self) -> bool:
        """Check whether there are parameters which require a matching simulation."""
        self._update()

        return len(self._require_sim_idx()) > 0

    def _add_sim(self, i, x):
        for k, v in x.items():
            self.x[k][i] = v
        self.m[i] = False

    def simulate(self, simulator: Callable):
        """Run simulator sequentially on parameter cache with missing corresponding simulations.

        Args:
            simulator: simulates an observation given a parameter input
        """
        self._update()

        idx = self._require_sim_idx()
        if len(idx) == 0:
            print("No simulations required.")
            return
        for i in tqdm(idx, desc="Simulate"):
            z = {k: v[i] for k, v in self.z.items()}
            x = simulator(z)
            self._add_sim(i, x)


class DirectoryCache(Cache):
    def __init__(self, zdim: int, xshape: Shape, path: PathType):
        """Instantiate an iP3 cache stored in a directory.

        Args:
            zdim: Number of z dimensions
            xshape: Shape of x array
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
    def __init__(self, param_names: int, obs_shapes: Shape, store=None):
        """Instantiate an iP3 cache stored in the memory.

        Args:
            zdim: Number of z dimensions
            obs_shapes: Shape of x array
            store (zarr.MemoryStore, zarr.DirectoryStore): optional, used in loading.
        """
        if store is None:
            self.store = zarr.MemoryStore()
        else:
            self.store = store
        super().__init__(param_names=param_names, obs_shapes=obs_shapes, store=self.store)

    def save(self, path: PathType) -> None:
        """Save the current state of the MemoryCache to a directory."""
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
        """Load existing DirectoryStore state into a MemoryCache object."""
        memory_store = zarr.MemoryStore()
        directory_store = zarr.DirectoryStore(path)
        zarr.convenience.copy_store(source=directory_store, dest=memory_store)

        group = zarr.group(store=memory_store)
        xshape = cls._extract_xshape_from_zarr_group(group)
        zdim = cls._extract_zdim_from_zarr_group(group)
        return MemoryCache(zdim=zdim, xshape=xshape, store=memory_store)


if __name__ == "__main__":
    pass
