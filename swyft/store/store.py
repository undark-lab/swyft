# pylint: disable=no-member, not-callable
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import fasteners
import numcodecs
import numpy as np
import zarr

import swyft
from swyft.store.simulator import SimulationStatus, Simulator
from swyft.types import PathType
from swyft.utils import is_empty


class Filesystem:
    metadata = "metadata"
    log_lambdas = "metadata/log_lambdas"
    samples = "samples"
    sims = "samples/sims"
    pars = "samples/pars"
    log_w = "samples/log_w"
    simulation_status = "samples/simulation_status"


class Store(ABC):
    """Abstract base class for various stores."""

    _filesystem = Filesystem

    @abstractmethod
    def __init__(
        self,
        params: Union[int, list],
        zarr_store: Union[zarr.MemoryStore, zarr.DirectoryStore],
        simulator: Optional[Simulator] = None,
        sync_path: Optional[PathType] = None,
        chunksize: int = 1000,
    ):
        """Initialize Store content dimensions.

        Args:
            params (list of strings or int): List of paramater names.  If int use ['z0', 'z1', ...].
            zarr_store: zarr storage.
            simulator: simulator object.
            sync_path: if specified, it will enable synchronization using file locks (files will be
                stored in the given path). Must be accessible to all processes working on the store
                and the underlying filesystem must support file locking.
            chunksize: the parameters and simulation output will be stored as arrays with the
                specified chunk size along the sample dimension (a single chunk will be used for the
                other dimensions).
        """
        self._zarr_store = zarr_store
        self._simulator = simulator

        if isinstance(params, int):
            params = ["z%i" % i for i in range(params)]
        self.params = params

        synchronizer = zarr.ProcessSynchronizer(sync_path) if sync_path else None
        self._root = zarr.group(store=self.zarr_store, synchronizer=synchronizer)

        logging.debug("  params = %s" % str(params))

        if set(["samples", "metadata"]) == set(self._root.keys()):
            logging.info("Loading existing store.")
            self._update()
        elif len(self._root.keys()) == 0:
            logging.info("Creating new store.")
            self._setup_new_zarr_store(
                len(self.params), simulator.sim_shapes, self._root, chunksize=chunksize
            )
            logging.debug("  sim_shapes = %s" % str(simulator.sim_shapes))
        else:
            raise KeyError(
                "The zarr storage is corrupted. It should either be empty or only have the keys ['samples', 'metadata']."
            )

        # a second layer of synchronization is required to grow the store
        self._lock = None
        if sync_path is not None:
            self._setup_lock(sync_path)

    @property
    def zarr_store(self):
        return self._zarr_store

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

    def _setup_new_zarr_store(
        self, zdim, sim_shapes, root, chunksize=1000
    ) -> None:  # Adding observational shapes to store
        # Parameters
        root.zeros(
            self._filesystem.pars,
            shape=(0, zdim),
            chunks=(chunksize, zdim),
            dtype="f8",
        )

        # Simulations
        sims = root.create_group(self._filesystem.sims)
        for name, shape in sim_shapes.items():
            sims.zeros(name, shape=(0, *shape), chunks=(chunksize, *shape), dtype="f8")

        # Random intensity weights
        root.zeros(
            self._filesystem.log_w,
            shape=(0,),
            chunks=(chunksize,),
            dtype="f8",
        )

        # Pickled Intensity (prior * N) objects
        root.create(
            self._filesystem.log_lambdas,
            shape=(0,),
            dtype=object,
            object_codec=numcodecs.Pickle(),
        )

        # Simulation status code
        root.zeros(
            self._filesystem.simulation_status,
            shape=(0,),
            chunks=(chunksize,),
            dtype="int",
        )

    def _update(self):
        self.sims = self._root[self._filesystem.sims]
        self.pars = self._root[self._filesystem.pars]
        self.log_w = self._root[self._filesystem.log_w]
        self.log_lambdas = self._root[self._filesystem.log_lambdas]
        self.sim_status = self._root[self._filesystem.simulation_status]

    def __len__(self):
        """Returns number of samples in the store."""
        self._update()
        return len(self.pars)

    def __getitem__(self, i):
        self._update()
        sim = {}
        for key, value in self.sims.items():
            sim[key] = value[i]
        par = self.pars[i]
        return (sim, par)

    def _append_new_points(self, pars, log_w):
        """Append z to zarr_store content and generate new slots for x."""
        self._update()
        n = len(pars)
        for key, value in self.sims.items():
            shape = list(value.shape)
            shape[0] += n
            value.resize(*shape)

        self._root[self._filesystem.pars].append(pars)
        self._root[self._filesystem.log_w].append(log_w)
        m = np.full(n, SimulationStatus.PENDING, dtype="int")
        self.sim_status.append(m)

    def log_lambda(self, z: np.ndarray) -> np.ndarray:
        self._update()
        d = -np.inf * np.ones_like(z[:, 0])
        if len(self.log_lambdas) == 0:
            return d
        for i in range(len(self.log_lambdas)):
            pdf = swyft.Prior.from_state_dict(self.log_lambdas[i]["pdf"])
            N = self.log_lambdas[i]["N"]
            r = pdf.log_prob(z) + np.log(N)
            d = np.where(r > d, r, d)
        return d

    def sample(self, N, pdf):

        # Lock store while adding new points
        self.lock()
        self._update()

        # Generate new points
        z_prop = pdf.sample(N=np.random.poisson(N))
        log_lambda_target = pdf.log_prob(z_prop) + np.log(N)
        log_lambda_store = self.log_lambda(z_prop)
        log_w = np.log(np.random.rand(len(z_prop))) + log_lambda_target
        accept_new = log_w > log_lambda_store
        z_new = z_prop[accept_new]
        log_w_new = log_w[accept_new]

        # Anything new?
        if sum(accept_new) > 0:
            # Add new entries to store
            self._append_new_points(z_new, log_w_new)
            logging.info(
                "  adding %i new samples to simulator store." % sum(accept_new)
            )
            # Update intensity function
            self.log_lambdas.resize(len(self.log_lambdas) + 1)
            self.log_lambdas[-1] = dict(pdf=pdf.state_dict(), N=N)

        # Points added, unlock store
        self.unlock()

        self._update()

        # Select points from cache
        z_store = self.pars
        log_w_store = self.log_w
        log_lambda_target = pdf.log_prob(z_store) + np.log(N)
        accept_stored = log_w_store < log_lambda_target
        indices = np.array(range(len(accept_stored)))[accept_stored]

        return indices

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
        current_status = self.sim_status.oindex[indices]
        if np.any(current_status == status):
            logging.warning(
                f"Changing simulation status to {status}, but some simulations have already status {status}"
            )
        self.sim_status.oindex[indices] = status

    def get_simulation_status(self, indices=None):
        """
        Determine the status of sample simulations.

        Args:
            indices: list of indices. If None, check the status of all samples

        Returns:
            list of simulation statuses
        """
        self._update()
        return (
            self.sim_status.oindex[indices]
            if indices is not None
            else self.sim_status[:]
        )

    def requires_sim(self, indices=None) -> bool:
        """Check whether there are parameters which require a matching simulation."""
        self._update()
        return self._get_indices_to_simulate(indices).size > 0

    def _get_indices_failed_simulations(self):
        self._update()
        return np.flatnonzero(self.sim_status == SimulationStatus.FAILED)

    @property
    def any_failed(self) -> bool:
        """Check whether there are parameters which currently lead to a failed simulation."""
        self._update()
        return self._get_indices_failed_simulations().size > 0

    def _add_sim(self, i, x):
        for k, v in x.items():
            self.sims[k][i] = v
        self._set_simulation_status(i, SimulationStatus.FINISHED)

    def _failed_sim(self, i):
        self._update()
        self._set_simulation_status(i, SimulationStatus.FAILED)

    def set_simulator(self, simulator):
        if self._simulator is not None:
            logging.warning("Simulator already set!  Overwriting.")
        self._simulator = simulator

    def simulate(
        self,
        indices: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
        wait_for_results: Optional[bool] = True,
    ) -> None:
        """Run simulator sequentially on parameter store with missing corresponding simulations.

        Args:
            indices: list of sample indices for which a simulation is required
            batch_size: simulations will be submitted in batches of the specified
                size
            wait_for_results: if True, return only when all simulations are done
        """
        if self._simulator is None:
            logging.warning("No simulator specified.  No simulations will run.")
            return

        self.lock()
        self._update()
        idx = self._get_indices_to_simulate(indices)
        self._set_simulation_status(idx, SimulationStatus.RUNNING)
        self.unlock()

        # Run simulations and collect status
        if len(idx) == 0:
            logging.debug("No simulations required.")
        else:
            # For the MemoryStore, we need to collect results in memory
            collect_in_memory = True if isinstance(self, MemoryStore) else False

            if collect_in_memory and not wait_for_results:
                logging.warning(
                    "Asynchronous collection of results is not implemented with the MemoryStore"
                )

            self._simulator.run(
                pars=self.pars,
                sims={k: v.oindex for k, v in self.sims.items()},
                sim_status=self.sim_status.oindex,
                indices=idx,
                collect_in_memory=collect_in_memory,
                batch_size=batch_size,
            )

        if wait_for_results:
            self.wait_for_simulations(indices)

    def wait_for_simulations(self, indices):
        """
        Wait for a set of sample simulations to be finished.
        Args:
            indices: list of sample indices
        """
        done = False
        while not done:
            time.sleep(1)
            status = self.get_simulation_status(indices)
            done = np.isin(status, [SimulationStatus.FINISHED, SimulationStatus.FAILED])
            done = np.all(done)


class DirectoryStore(Store):
    def __init__(
        self,
        params,
        path: PathType,
        sync_path: Optional[PathType] = None,
        simulator=None,
    ):
        """Instantiate an iP3 store stored in a directory.

        Args:
            params (list of strings or int): List of paramater names.  If int use ['z0', 'z1', ...].
            path: path to storage directory
            sync_path: path for synchronization via file locks (files will be stored in the given path).
                It must differ from path, it must be accessible to all processes working on the store,
                and the underlying filesystem must support file locking.
            simulator: simulator object.
        """
        zarr_store = zarr.DirectoryStore(path)
        sync_path = sync_path or os.path.splitext(path)[0] + ".sync"
        super().__init__(
            params=params,
            zarr_store=zarr_store,
            sync_path=sync_path,
            simulator=simulator,
        )

    @classmethod
    def load(cls, path: PathType):
        """Load existing DirectoryStore."""
        zarr_store = zarr.DirectoryStore(path)
        group = zarr.group(store=zarr_store)
        zdim = group[cls._filesystem.pars].shape[1]
        return DirectoryStore(params=zdim, path=path)


class MemoryStore(Store):
    def __init__(self, params, zarr_store=None, simulator=None):
        """Instantiate an iP3 store stored in the memory.

        Args:
            params (list of strings or int): List of paramater names.  If int use ['z0', 'z1', ...].
            zarr_store (zarr.MemoryStore, zarr.DirectoryStore): optional, used in
                loading.
            simulator: simulator object.
        """
        if zarr_store is None:
            zarr_store = zarr.MemoryStore()
            logging.debug("Creating new empty MemoryStore.")
        else:
            logging.debug("Creating MemoryStore from zarr_store.")
        super().__init__(params=params, zarr_store=zarr_store, simulator=simulator)

    def save(self, path: PathType) -> None:
        """Save the current state of the MemoryStore to a directory."""
        path = Path(path)
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"{path} should be a directory")
        elif path.exists() and not is_empty(path):
            raise FileExistsError(f"{path} is not empty")
        else:
            path.mkdir(parents=True, exist_ok=True)
            zarr_store = zarr.DirectoryStore(path)
            zarr.convenience.copy_store(source=self.zarr_store, dest=zarr_store)
            return None

    @classmethod
    def load(cls, path: PathType):
        """Load existing DirectoryStore state into a MemoryStore object."""
        memory_store = zarr.MemoryStore()
        directory_store = zarr.DirectoryStore(path)
        zarr.convenience.copy_store(source=directory_store, dest=memory_store)

        group = zarr.group(store=memory_store)
        zdim = group[cls._filesystem.pars].shape[1]
        return MemoryStore(params=zdim, zarr_store=memory_store)

    @classmethod
    def from_model(cls, model, prior):
        """Convenience function to instantiate new MemoryStore with given model and prior.

        Args:
            model (function): Simulator model.
            prior (Prior): Model prior.

        Note:
            The simulator model is run once in order to infer observable shapes from the output.
        """
        v = prior.sample(1)[0]
        vdim = len(v)
        sim = model(v)
        sim_shapes = {k: v.shape for k, v in sim.items()}
        simulator = swyft.Simulator(model, sim_shapes=sim_shapes)
        return MemoryStore(vdim, simulator=simulator)
