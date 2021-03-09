# pylint: disable=no-member, not-callable
import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Callable, Dict, Union
from warnings import warn

import numcodecs
import numpy as np
import zarr
from tqdm import tqdm

from swyft.cache.exceptions import LowIntensityError
from swyft.ip3 import Intensity
from swyft.types import Array, PathType, Shape
from swyft.utils import all_finite, is_empty, verbosity

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

        logging.debug("Creating Cache.")
        logging.debug("  params = %s" % str(params))
        logging.debug("  obs_shapes = %s" % str(obs_shapes))

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
            # FIX: Too small chunks lead to problems with appending

        # Adding observational shapes to store
        x = root.create_group(self._filesystem.obs)
        for name, shape in obs_shapes.items():
            x.zeros(name, shape=(0, *shape), chunks=(1, *shape), dtype="f8")

        # Requires simulation flag
        m = root.zeros(  # noqa: F841
            self._filesystem.requires_simulation,
            shape=(0, 1),
            chunks=(100000, 1),
            dtype="bool",
        )

        # Failed simulation flag
        f = root.zeros(  # noqa: F841
            self._filesystem.failed_simulation,
            shape=(0, 1),
            chunks=(100000, 1),
            dtype="bool",
        )

        # Which intensity flag
        wu = self.root.zeros(  # noqa: F841
            self._filesystem.which_intensity,
            shape=(0,),
            chunks=(100000,),
            dtype="i4",
        )

        # Intensity object
        u = root.create(  # noqa: F841
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
        assert len(self.f) == len(
            self.m
        ), "Metadata noting which indices require simulation and which have failed have desynced."
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
        assert len(self.u) == len(
            self.intensities
        ), "The intensity pickles should be the same length as the state dicts."
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
        wu = self.intensity_len * np.ones(n, dtype="int")
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

    def grow(self, prior: "swyft.intensity.Intensity", N):  # noqa
        """Given an intensity function, add parameter samples to the cache.

        Args:
            intensity: target parameter intensity function
        """
        intensity = Intensity(prior, N)

        # Proposed new samples z from p
        z_prop = intensity.sample()

        # Rejection sampling from proposal list
        accepted = []
        cached_log_intensities = self.intensity(z_prop)
        target_log_intensities = intensity(z_prop)
        for cached_log_intensity, target_log_intensity in zip(
            cached_log_intensities, target_log_intensities
        ):
            log_prob_reject = np.minimum(0, cached_log_intensity - target_log_intensity)
            accepted.append(log_prob_reject < np.log(np.random.rand()))
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

    def sample(self, prior: "swyft.intensity.Intensity", N):  # noqa
        """Sample from Cache.

        Args:
            intensity: target parameter intensity function
        """
        intensity = Intensity(prior, N)

        self._update()

        # self.grow(prior, N)

        accepted = []
        zlist = {k: self.z[k][:] for k in (self.z)}

        cached_intensities = self.intensity(zlist)
        target_intensities = intensity(zlist)
        assert len(self) == len(cached_intensities)
        assert len(self) == len(target_intensities)
        for i, (target_intensity, cached_intensity) in enumerate(
            zip(target_intensities, cached_intensities)
        ):
            log_prob_accept = target_intensity - cached_intensity
            if log_prob_accept > 0.0:
                raise LowIntensityError(
                    f"{log_prob_accept} > 0, "
                    " but we expected the log ratio of target intensity function to the cache <= 0. "
                    "There may not be enough samples in the cache or "
                    "a constrained intensity function was not accounted for."
                )
            elif log_prob_accept > np.log(np.random.rand()):
                accepted.append(i)
            else:
                continue
        return accepted

    def _get_idx_requiring_sim(self):
        indices = []
        m = self.m[:]
        for i in range(len(self)):
            if m[i]:
                indices.append(i)
        return indices

    @property
    def requires_sim(self) -> bool:
        """Check whether there are parameters which require a matching simulation."""
        self._update()
        return len(self._get_idx_requiring_sim()) > 0

    def _get_idx_failing_sim(self):
        indices = []
        f = self.f[:]
        for i, f in enumerate(f):
            if f:
                indices.append(i)
        return indices

    @property
    def any_failed(self) -> bool:
        """Check whether there are parameters which currently lead to a failed simulation."""
        self._update()
        return len(self._get_idx_failing_sim()) > 0

    def _add_sim(self, i, x):
        for k, v in x.items():
            self.x[k][i] = v
        self.m[i] = False
        self.f[i] = False

    def _failed_sim(self, i):
        self.f[i] = True

    def _replace(self, i, z, x):
        for key, value in z.items():
            self.z[key][i] = value
        for k, v in x.items():
            self.x[k][i] = v
        self.m[i] = False
        self.f[i] = False

    @staticmethod
    def did_simulator_succeed(x: Dict[str, Array], fail_on_non_finite: bool) -> bool:
        """Is the simulation a success?"""

        assert isinstance(x, dict), "Simulators must return a dictionary."

        dict_anynone = lambda d: any(v is None for v in d.values())

        if dict_anynone(x):
            return False
        elif fail_on_non_finite and not all_finite(x):
            return False
        else:
            return True

    def simulate(
        self,
        simulator: Callable,
        fail_on_non_finite: bool = True,
        max_attempts: int = 1000,
    ) -> None:
        """Run simulator sequentially on parameter cache with missing corresponding simulations.

        Args:
            simulator: simulates an observation given a parameter input
            fail_on_non_finite: if nan / inf in simulation, considered a failed simulation
            max_attempts: maximum number of resample attempts before giving up.
        """
        self._update()

        idx = self._get_idx_requiring_sim()
        if len(idx) == 0:
            if verbosity() >= 2:
                print("No simulations required.")
            return True
        for i in tqdm(idx, desc="Simulate"):
            z = {k: v[i] for k, v in self.z.items()}
            x = simulator(z)
            success = self.did_simulator_succeed(x, fail_on_non_finite)
            if success:
                self._add_sim(i, x)
            else:
                self._failed_sim(i)

        if self.any_failed:
            self.resample_failed_simulations(
                simulator, fail_on_non_finite, max_attempts
            )
            # TODO add test which ensures that volume does not change upon more samples.

        if self.any_failed:
            warn(
                f"Some simulations failed, despite {max_attempts} to resample them. They have been marked in the cache."
            )

    def resample_failed_simulations(
        self, simulator: Callable, fail_on_non_finite: bool, max_attempts: int
    ) -> None:
        self._update
        if self.any_failed:
            idx = self._get_idx_failing_sim()
            for i in tqdm(idx, desc="Fix failed sims"):
                iters = 0
                success = False
                which_intensity = self.wu[i]
                prior = self.intensities[which_intensity].prior
                while not success and iters < max_attempts:
                    param = prior.sample(1)
                    z = {k: v[0] for k, v in param.items()}
                    x = simulator(z)
                    success = self.did_simulator_succeed(x, fail_on_non_finite)
                    iters += 1
                if success:
                    self._replace(i, z, x)
            return None
        else:
            if verbosity() >= 2:
                print("No failed simulations.")
            return None


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
            logging.debug("Creating new empty MemoryCache.")
        else:
            self.store = store
            logging.debug("Creating MemoryCache from store.")
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
        params = {k: v.item() for k, v in params.items()}
        obs = model(params)
        obs_shapes = {k: v.shape for k, v in obs.items()}

        return MemoryCache(list(prior.prior_config.keys()), obs_shapes)
