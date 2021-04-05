# pylint: disable=no-member, not-callable
import enum
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from warnings import warn

import fasteners
import numcodecs
import numpy as np
import zarr

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
        "simulation_status",
        "failed_simulation",
        "which_intensity",
    ],
)


class SimulationStatus(enum.IntEnum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2


class Cache(ABC):
    """Abstract base class for various caches."""

    _filesystem = Filesystem(
        "metadata",
        "metadata/intensity",
        "samples",
        "samples/obs",
        "samples/par",
        "samples/simulation_status",
        "samples/failed_simulation",
        "samples/which_intensity",
    )

    @abstractmethod
    def __init__(
        self,
        params,
        obs_shapes: Shape,
        store: Union[zarr.MemoryStore, zarr.DirectoryStore],
        sync_path: Optional[PathType] = None,
    ):
        """Initialize Cache content dimensions.

        Args:
            params (list of strings): List of paramater names
            obs_shapes (dict): Map of obs names to shapes
            store: zarr storage.
            sync_path: path to the cache lock files. Must be accessible to all
                processes working on the cache.
        """
        self.store = store
        self.params = params
        synchronizer = (
            None if sync_path is None else zarr.ProcessSynchronizer(sync_path)
        )
        self.root = zarr.group(store=self.store, synchronizer=synchronizer)
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
        self._lock = None
        if sync_path is not None:
            self._setup_lock(sync_path)
        # assert (
        #    zdim == self.zdim
        # ), f"Your given zdim, {zdim}, was not equal to the one defined in zarr {self.zdim}."
        # assert (
        #    xshape == self.xshape
        # ), f"Your given xshape, {xshape}, was not equal to the one defined in zarr {self.xshape}."

    def _setup_lock(self, sync_path):
        path = os.path.join(sync_path, "cache.lock")
        self._lock = fasteners.InterProcessLock(path)

    def lock(self):
        if self._lock is not None:
            logging.debug("Cache locked")
            self._lock.acquire(blocking=True)

    def unlock(self):
        if self._lock is not None:
            self._lock.release()
            logging.debug("Cache unlocked")

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

        # Simulation status code
        m = root.zeros(  # noqa: F841
            self._filesystem.simulation_status,
            shape=(0,),
            chunks=(100000,),
            dtype="int",
        )

        # Failed simulation flag
        f = root.zeros(  # noqa: F841
            self._filesystem.failed_simulation,
            shape=(0,),
            chunks=(100000,),
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
    def _extract_obs_shapes_from_zarr_group(group):
        obs_key = [k for k in group[Cache._filesystem.obs].keys()]
        assert len(obs_key) == 1
        obs_key = obs_key[0]
        return group[Cache._filesystem.obs][obs_key].shape[1:]

    @staticmethod
    def _extract_params_from_zarr_group(group):
        return [k for k in group[Cache._filesystem.par].keys()]

    def _update(self):
        # This could be removed with a property for each attribute which only loads from disk if something has changed. TODO
        self.x = self.root[self._filesystem.obs]
        self.z = self.root[self._filesystem.par]
        self.m = self.root[self._filesystem.simulation_status]
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

        # Register as pending
        m = np.full(n, SimulationStatus.PENDING, dtype="int")
        self.m.append(m)

        # Simulations have not failed, yet.
        f = np.zeros(n, dtype="bool")
        self.f.append(f)

        # Which intensity was a parameter drawn with
        wu = self.intensity_len * np.ones(n, dtype="int")
        self.wu.append(wu)

    def log_intensity(self, z: Dict[str, np.ndarray]) -> np.ndarray:
        self._update()

        # How many parameters are we evaluating on?
        shapes = [array.shape for array in z.values()]
        shape = shapes[0]
        assert all(shape == s for s in shapes)
        length = shape[0]

        if len(self.u) == 0:
            # An empty cache has log intensity of -infinity.
            return -np.inf * np.ones(length)
        else:
            return np.stack(
                [log_intensity(z) for log_intensity in self.intensities]
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
        cached_log_intensities = self.log_intensity(z_prop)
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
            logging.debug("No samples added to simulator cache")
            pass

        # save new intensity function. We collect them all to find their maximum.
        # NOTE: We only do this when new samples are added. This is not
        # entierly correct statistically, but a pain otherwise.
        if sum(accepted) > 0:
            self.u.resize(len(self.u) + 1)
            self.u[-1] = intensity.state_dict()
            self.intensities.append(intensity)

    def sample(
        self, prior: "swyft.intensity.Intensity", N: int
    ) -> List[int]:  # noqa: F821
        intensity = Intensity(prior, N)

        self.lock()
        self._update()
        self.grow(prior, N)
        self.unlock()

        zlist = {k: self.z[k][:] for k in (self.z)}

        cached_intensities = self.log_intensity(zlist)
        target_intensities = intensity(zlist)
        assert len(self) == len(cached_intensities)
        assert len(self) == len(target_intensities)
        log_prob_accept = target_intensities - cached_intensities
        if np.any(log_prob_accept > 0.0):
            raise LowIntensityError(
                "We expect the log ratio of target intensity function to the cache <= 0. "
                "There may not be enough samples in the cache or "
                "a constrained intensity function was not accounted for."
            )
        prob_reject = 1.0 - np.random.random(log_prob_accept.shape)  # (0;1]
        accept = log_prob_accept > np.log(prob_reject)
        accepted = np.flatnonzero(accept)
        return accepted

    def _get_indices_to_simulate(self, indices=None):
        """
        Determine which samples need to be simulated.

        Args:
            indices: (optional) array with the indices of the samples to
            consider. If None, consider all samples.

        Returns:
            array with the sample indices
        """
        status = self.get_simulation_status(indices)
        require_simulation = status == SimulationStatus.PENDING
        idx = np.flatnonzero(require_simulation)
        return indices[idx] if indices is not None else idx

    def _set_simulation_status(self, indices, status):
        """
        Flag the specified samples with the simulation status.

        Args:
            indices: array with the indices of the samples to flag
            status: new status for the samples
        """
        assert status in list(SimulationStatus), f"Unknown status {status}"
        current_status = self.m.oindex[indices]
        if np.any(current_status == status):
            raise ValueError(f"Some simulations have already status {status}")
        self.m.oindex[indices] = status

    def get_simulation_status(self, indices=None):
        """
        Determine the status of sample simulations.

        Args:
            indices: list of indices. If None, check the status of all samples

        Returns:
            list of simulation statuses
        """
        self._update()
        return self.m.oindex[indices] if indices is not None else self.m[:]

    @property
    def requires_sim(self) -> bool:
        """Check whether there are parameters which require a matching simulation."""
        self._update()
        return self._get_indices_to_simulate().size > 0

    def _get_indices_failed_simulations(self):
        return np.flatnonzero(self.f[:])

    @property
    def any_failed(self) -> bool:
        """Check whether there are parameters which currently lead to a failed simulation."""
        self._update()
        return self._get_indices_failed_simulations().size > 0

    def _add_sim(self, i, x):
        for k, v in x.items():
            self.x[k][i] = v
        self._set_simulation_status(i, SimulationStatus.FINISHED)
        self.f[i] = False

    def _failed_sim(self, i):
        self.f[i] = True

    def _replace(self, i, z, x):
        for key, value in z.items():
            self.z[key][i] = value
        for k, v in x.items():
            self.x[k][i] = v
        self._set_simulation_status(i, SimulationStatus.FINISHED)
        self.f[i] = False

    def simulate(
        self,
        simulator,
        indices: Optional[List[int]] = None,
    ) -> None:
        """Run simulator sequentially on parameter cache with missing corresponding simulations.

        Args:
            simulator: simulates an observation given a parameter input
            indices: list of sample indices for which a simulation is required
            fail_on_non_finite: if nan / inf in simulation, considered a failed simulation
        """
        self.lock()
        idx = self._get_indices_to_simulate(indices)
        self._set_simulation_status(idx, SimulationStatus.RUNNING)
        self.unlock()

        if len(idx) == 0:
            if verbosity() >= 2:
                print("No simulations required.")
        else:
            z = [{k: v[i] for k, v in self.z.items()} for i in idx]
            res = simulator.run(z)
            x_all, validity = list(zip(*res))  # TODO: check other data formats

            for i, x, v in zip(idx, x_all, validity):
                if v == 0:
                    self._add_sim(i, x)
                else:
                    self._failed_sim(i)

        # some of the samples might be run by other processes - wait for these
        self.wait_for_simulations(indices)

    def wait_for_simulations(self, indices):
        """
        Wait for a set of sample simulations to be finished.

        Args:
            indices: list of sample indices
        """
        status = self.get_simulation_status(indices)
        while not np.all(status == SimulationStatus.FINISHED):
            time.sleep(1)
            status = self.get_simulation_status(indices)


class DirectoryCache(Cache):
    def __init__(
        self,
        params,
        obs_shapes: Shape,
        path: PathType,
        sync_path: Optional[PathType] = None,
    ):
        """Instantiate an iP3 cache stored in a directory.

        Args:
            zdim: Number of z dimensions
            obs_shapes: Shape of x array
            path: path to storage directory
            sync_path: path to the cache lock files. Must be accessible to all
                processes working on the cache and should differ from `path`.
        """
        self.store = zarr.DirectoryStore(path)
        sync_path = sync_path or os.path.splitext(path)[0] + ".sync"
        super().__init__(
            params=params, obs_shapes=obs_shapes, store=self.store, sync_path=sync_path
        )

    @classmethod
    def load(cls, path: PathType):
        """Load existing DirectoryStore."""
        store = zarr.DirectoryStore(path)
        group = zarr.group(store=store)
        obs_shapes = cls._extract_obs_shapes_from_zarr_group(group)
        z = cls._extract_params_from_zarr_group(group)
        return DirectoryCache(params=z, obs_shapes=obs_shapes, path=path)


class MemoryCache(Cache):
    def __init__(
        self,
        params,
        obs_shapes,
        store=None,
    ):
        """Instantiate an iP3 cache stored in the memory.

        Args:
            zdim: Number of z dimensions
            obs_shapes: Shape of x array
            store (zarr.MemoryStore, zarr.DirectoryStore): optional, used in
                loading.
        """
        if store is None:
            self.store = zarr.MemoryStore()
            logging.debug("Creating new empty MemoryCache.")
        else:
            self.store = store
            logging.debug("Creating MemoryCache from store.")
        super().__init__(
            params=params,
            obs_shapes=obs_shapes,
            store=self.store,
        )

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
        obs_shapes = cls._extract_obs_shapes_from_zarr_group(group)
        z = cls._extract_params_from_zarr_group(group)
        return MemoryCache(params=z, obs_shapes=obs_shapes, store=memory_store)

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
