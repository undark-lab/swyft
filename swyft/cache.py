# pylint: disable=no-member, not-callable
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from warnings import warn

import numcodecs
import numpy as np
import torch
import zarr
from scipy.interpolate import interp1d
from tqdm import tqdm

from .intensity import Intensity
from .types import Array, Callable, Dict, PathType, Shape, Union
from .utils import allfinite, is_empty, verbosity


class LowIntensityError(Exception):
    pass


class NormalizeStd:
    def __init__(self, values):
        self.mean = {}
        self.std = {}

        for k, v in values.items():
            self.mean[k] = v.mean(axis=0)
            self.std[k] = v.std(axis=0).mean()

    def __call__(self, values):
        out = {}
        for k, v in values.items():
            out[k] = (v - self.mean[k]) / self.std[k]
        return out


class NormalizeScale:
    def __init__(self, values):
        self.median = {}
        self.perc = {}

        for k, v in values.items():
            median = np.percentile(v, 50, axis=0)
            perc = np.percentile(v - median, np.linspace(0, 100, 101))
            self.median[k] = median
            self.perc[k] = perc

    def __call__(self, values):
        out = {}
        for k, v in values.items():
            v = v - self.median[k]
            v = interp1d(
                self.perc[k], np.linspace(-1, 1, 101), fill_value="extrapolate"
            )(v)
            out[k] = v
        return out


Normalize = NormalizeStd


class Transform:
    def __init__(self, par_combinations, param_transform=None, obs_transform=None):
        self.obs_transform = (lambda x: x) if obs_transform is None else obs_transform
        self.param_transform = (
            (lambda z: z) if param_transform is None else param_transform
        )
        self.par_combinations = par_combinations
        self.par_comb_shape = self._get_par_comb_shape(par_combinations)

    def _get_par_comb_shape(self, par_combinations):
        n = len(par_combinations)
        m = max([len(c) for c in par_combinations])
        return (n, m)

    def _combine(self, par):
        shape = par[list(par)[0]].shape
        if len(shape) == 0:
            out = torch.zeros(self.par_comb_shape)
            for i, c in enumerate(self.par_combinations):
                pars = torch.stack([par[k] for k in c]).T
                out[i, : pars.shape[0]] = pars
        else:
            n = shape[0]
            out = torch.zeros((n,) + self.par_comb_shape)
            for i, c in enumerate(self.par_combinations):
                pars = torch.stack([par[k] for k in c]).T
                out[:, i, : pars.shape[1]] = pars
        return out

    def _tensorfy(self, x):
        return {k: torch.tensor(v).float() for k, v in x.items()}

    def __call__(self, obs=None, par=None):
        out = {}
        if obs is not None:
            tmp = self.obs_transform(obs)
            out["obs"] = self._tensorfy(tmp)
        if par is not None:
            tmp = self.param_transform(par)
            z = self._tensorfy(tmp)
            out["par"] = self._combine(z)
        return out


class Dataset(torch.utils.data.Dataset):
    def __init__(self, points):
        self.points = points

    def _tensorfy(self, x):
        return {k: torch.tensor(v).float() for k, v in x.items()}

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        p = self.points[i]
        return dict(obs=self._tensorfy(p["obs"]), par=self._tensorfy(p["par"]))


Filesystem = namedtuple(
    "Filesystem",
    [
        "metadata",
        "intensity",
        "samples",
        "obs",
        "par",
        "requires_simulation",
        "failed_simulation",
        "which_intensity",
    ],
)


class Cache(ABC):
    """Abstract base class for various caches."""

    _filesystem = Filesystem(
        "metadata",
        "metadata/intensity",
        "samples",
        "samples/obs",
        "samples/par",
        "samples/requires_simulation",
        "samples/failed_simulation",
        "samples/which_intensity",
    )

    @abstractmethod
    def __init__(
        self,
        params,
        obs_shapes: Shape,
        store: Union[zarr.MemoryStore, zarr.DirectoryStore],
    ):
        """Initialize Cache content dimensions.

        Args:
            params (list of strings): List of paramater names
            obs_shapes (dict): Map of obs names to shapes
            store: zarr storage.
        """
        self.store = store
        self.params = params
        self.root = zarr.group(store=self.store)
        self.intensities = []

        if all(key in self.root.keys() for key in ["samples", "metadata"]):
            if verbosity() >= 1:
                print("Loading existing cache.")
            self._update()
        elif len(self.root.keys()) == 0:
            if verbosity() >= 1:
                print("Creating new cache.")
            self._setup_new_cache(params, obs_shapes, self.root)
        else:
            raise KeyError(
                "The zarr storage is corrupted. It should either be empty or only have the keys ['samples', 'metadata']."
            )

        # assert (
        #    zdim == self.zdim
        # ), f"Your given zdim, {zdim}, was not equal to the one defined in zarr {self.zdim}."
        # assert (
        #    xshape == self.xshape
        # ), f"Your given xshape, {xshape}, was not equal to the one defined in zarr {self.xshape}."

    def _setup_new_cache(self, params, obs_shapes, root) -> None:
        # Add parameter names to store
        z = root.create_group(self._filesystem.par)
        for name in params:
            z.zeros(name, shape=(0,), chunks=(100000,), dtype="f8")
            # FIX: Too mall chunks lead to problems with appending

        # Adding observational shapes to store
        x = root.create_group(self._filesystem.obs)
        for name, shape in obs_shapes.items():
            x.zeros(name, shape=(0, *shape), chunks=(1, *shape), dtype="f8")

        # Requires simulation flag
        m = root.zeros(
            self._filesystem.requires_simulation,
            shape=(0, 1),
            chunks=(100000, 1),
            dtype="bool",
        )

        # Failed simulation flag
        f = root.zeros(
            self._filesystem.failed_simulation,
            shape=(0, 1),
            chunks=(100000, 1),
            dtype="bool",
        )

        # Which intensity flag
        wu = self.root.zeros(
            self._filesystem.which_intensity,
            shape=(0, 1),
            chunks=(100000, 1),
            dtype="i4",
        )

        # Intensity object
        u = root.create(
            self._filesystem.intensity,
            shape=(0,),
            dtype=object,
            object_codec=numcodecs.Pickle(),
        )

    @staticmethod
    def _extract_xshape_from_zarr_group(group):
        return group[Cache._filesystem.obs].shape[1:]

    @staticmethod
    def _extract_zdim_from_zarr_group(group):
        return group[Cache._filesystem.par].shape[1]

    def _update(self):
        # This could be removed with a property for each attribute which only loads from disk if something has changed. TODO
        self.x = self.root[self._filesystem.obs]
        self.z = self.root[self._filesystem.par]
        self.m = self.root[self._filesystem.requires_simulation]
        self.f = self.root[self._filesystem.failed_simulation]
        self.u = self.root[self._filesystem.intensity]
        self.wu = self.root[self._filesystem.which_intensity]
        self.intensities = [
            Intensity.from_state_dict(self.u[i]) for i in range(len(self.u))
        ]

    def __len__(self):
        """Returns number of samples in the cache."""
        self._update()
        # Return len of first entry
        params = list(self.z)
        return len(self.z[params[0]])

    @property
    def intensity_len(self):
        self._update()
        return len(self.u)

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
            value.append(z[key])

        # Register as missing
        m = np.ones((n, 1), dtype="bool")
        self.m.append(m)

        # Simulations have not failed, yet.
        self.f.append(~m)

        # Which intensity was a parameter drawn with
        wu = self.intensity_len * m
        self.wu.append(wu)

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
            return np.array(
                [self.intensities[i](z) for i in range(len(self.intensities))]
            ).max(axis=0)

    def grow(self, prior: "swyft.intensity.Intensity", N):
        """Given an intensity function, add parameter samples to the cache.

        Args:
            intensity: target parameter intensity function
        """
        intensity = Intensity(prior, N)

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
            if verbosity() >= 1:
                print("  adding %i new samples to simulator cache." % sum(accepted))
        else:
            pass

        # save new intensity function. We collect them all to find their maximum.
        # NOTE: We only do this when new samples are added. This is not
        # entierly correct statistically, but a pain otherwise.
        if sum(accepted) > 0:
            self.u.resize(len(self.u) + 1)
            self.u[-1] = intensity.state_dict()
            self.intensities.append(intensity)

    def sample(self, prior: "swyft.intensity.Intensity", N):
        """Sample from Cache.

        Args:
            intensity: target parameter intensity function
        """
        intensity = Intensity(prior, N)

        self._update()

        # self.grow(prior, N)

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
        self.f[i] = False

    def _failed_sim(self, i):
        self.f[i] = True

    def did_simulator_succeed(
        self, x: Dict[str, Array], fail_on_non_finite: bool
    ) -> bool:
        """Is the simulation a success?"""

        assert isinstance(x, dict), "Simulators must return a dictionary."

        dict_anynone = lambda d: any(v is None for v in d.values())
        dict_allfinite = lambda d: all(allfinite(v) for v in d.values())

        if dict_anynone(x):
            return False
        elif fail_on_non_finite and not dict_allfinite(x):
            return False
        else:
            return True

    def simulate(self, simulator: Callable, fail_on_non_finite: bool = True) -> bool:
        """Run simulator sequentially on parameter cache with missing corresponding simulations.

        Args:
            simulator: simulates an observation given a parameter input
            fail_on_non_finite: if nan / inf in simulation, considered a failed simulation
        """
        self._update()
        success = True

        idx = self._require_sim_idx()
        if len(idx) == 0:
            if verbosity() >= 2:
                print("No simulations required.")
            return
        for i in tqdm(idx, desc="Simulate"):
            z = {k: v[i] for k, v in self.z.items()}
            x = simulator(z)
            success = self.did_simulator_succeed(x, fail_on_non_finite)
            if success:
                self._add_sim(i, x)
            else:
                self._failed_sim(i)

        if not success:
            warn("Some simulations failed. They have been marked.")
        return success  # TODO functionality to deal failed simulations automatically


class DirectoryCache(Cache):
    def __init__(self, params, obs_shapes: Shape, path: PathType):
        """Instantiate an iP3 cache stored in a directory.

        Args:
            zdim: Number of z dimensions
            xshape: Shape of x array
            path: path to storage directory
        """
        self.store = zarr.DirectoryStore(path)
        super().__init__(params=params, obs_shapes=obs_shapes, store=self.store)

    @classmethod
    def load(cls, path: PathType):
        """Load existing DirectoryStore."""
        store = zarr.DirectoryStore(path)
        group = zarr.group(store=store)
        xshape = cls._extract_xshape_from_zarr_group(group)
        zdim = cls._extract_zdim_from_zarr_group(group)
        return DirectoryCache(zdim=zdim, xshape=xshape, path=path)


class MemoryCache(Cache):
    def __init__(self, params, obs_shapes, store=None):
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
        super().__init__(params=params, obs_shapes=obs_shapes, store=self.store)

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

    @classmethod
    def from_simulator(cls, model, prior):
        """Convenience function to instantiate new MemoryCache with correct obs_shapes.

        Args:
            model (function): Simulator model.
            prior (Prior): Model prior.

        Note:
            The simulator model is run once in order to infer observable shapes from the output.
        """
        params = prior.sample(1)
        params = {k: v[0] for k, v in params.items()}
        obs = model(params)
        obs_shapes = {k: v.shape for k, v in obs.items()}

        return MemoryCache(list(prior.prior_conf.keys()), obs_shapes)


if __name__ == "__main__":
    pass
