# pylint: disable=no-member, not-callable
import logging
import os
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import fasteners
import numcodecs
import numpy as np
import zarr

import swyft
from swyft.store.simulator import SimulationStatus, Simulator
from swyft.types import Array, ObsShapeType, ParameterNamesType, PathType
from swyft.utils import is_empty

log = logging.getLogger(__name__)


class Filesystem:
    metadata = "metadata"
    log_lambdas = "metadata/log_lambdas"
    samples = "samples"
    sims = "samples/sims"
    v = "samples/v"
    log_w = "samples/log_w"
    simulation_status = "samples/simulation_status"


class Store:
    """Store of sample parameters and simulation outputs.

    Based on Zarr, it should be instantiated via its methods `memory_store`,
    `directory_store` or `load`.

    Args:
        zarr_store: Zarr store object.
        simulator: simulator object.
        sync_path: if specified, it will enable synchronization using file locks (files will be
            stored in the given path). Must be accessible to all processes working on the store
            and the underlying filesystem must support file locking.
        chunksize: the parameters and simulation output will be stored as arrays with the
            specified chunk size along the sample dimension (a single chunk will be used for the
            other dimensions).
        pickle_protocol: pickle protocol number used for storing intensity functions.
        from_scratch: if False, load the sample store from the Zarr store provided.
    """

    _filesystem = Filesystem

    def __init__(
        self,
        zarr_store: Union[zarr.MemoryStore, zarr.DirectoryStore],
        simulator: Optional[Simulator] = None,
        sync_path: Optional[PathType] = None,
        chunksize: int = 1,
        pickle_protocol: int = 4,
        from_scratch: bool = True,
    ) -> None:
        self._zarr_store = zarr_store
        self._simulator = simulator
        self._pickle_protocol = pickle_protocol  # TODO: to be deprecated, we will default to 4, which is supported since python 3.4

        synchronizer = zarr.ProcessSynchronizer(sync_path) if sync_path else None
        self._root = zarr.group(
            store=self._zarr_store, synchronizer=synchronizer, overwrite=from_scratch
        )

        if not from_scratch:
            if not {"samples", "metadata"} == self._root.keys():
                raise KeyError(
                    "Invalid Zarr store. It should have keys: ['samples', 'metadata']."
                )
            print("Loading existing store.")
            self._update()
        else:
            print("Creating new store.")
            if simulator is None:
                raise ValueError("A simulator is required to setup a new store.")
            self._setup_new_zarr_store(
                simulator.parameter_names,
                simulator.sim_shapes,
                self._root,
                chunksize=chunksize,
                sim_dtype=simulator.sim_dtype,
            )
            log.debug("  sim_shapes = %s" % str(simulator.sim_shapes))

        # a second layer of synchronization is required to grow the store
        self._lock = None
        if sync_path is not None:
            self._setup_lock(sync_path)

    def add(
        self, N: int, prior: "swyft.Prior", bound: Optional["swyft.Bound"] = None
    ) -> None:
        """Adds points to the store.

        Args:
            N: Number of samples
            prior: Prior
            bound: Bound object for prior truncation

        .. warning::
            Calling this method will alter the content of the store by adding
            additional points. Currently this cannot be reverted, so use with
            care when applying it to the DirectoryStore.
        """
        pdf = swyft.PriorTruncator(prior, bound)

        # Lock store while adding new points
        self.lock()
        self._update()

        # Generate new points
        v_prop = pdf.sample(np.random.poisson(N))
        log_lambda_target = pdf.log_prob(v_prop) + np.log(N)
        log_lambda_store = self.log_lambda(v_prop)
        log_w = np.log(np.random.rand(len(v_prop))) + log_lambda_target
        accept_new = log_w > log_lambda_store
        v_new = v_prop[accept_new]
        log_w_new = log_w[accept_new]

        # Anything new?
        if sum(accept_new) > 0:
            # Add new entries to store
            self._append_new_points(v_new, log_w_new)
            print("Store: Adding %i new samples to simulator store." % sum(accept_new))
            # Update intensity function
            self.log_lambdas.resize(len(self.log_lambdas) + 1)
            self.log_lambdas[-1] = dict(pdf=pdf.state_dict(), N=N)

        log.debug(f"  total size of simulator store {len(self)}.")

        # Points added, unlock store
        self.unlock()
        self._update()

    #    @property
    #    def zarr_store(self):
    #        """Return ZarrStore object."""
    #        return self._zarr_store

    def _setup_lock(self, sync_path: PathType) -> None:
        """Setup lock for concurrent access from multiple processes."""
        path = os.path.join(sync_path, "cache.lock")
        self._lock = fasteners.InterProcessLock(path)

    def lock(self) -> None:
        """Lock store for the current process."""
        if self._lock is not None:
            log.debug("Store locked")
            self._lock.acquire(blocking=True)

    def unlock(self) -> None:
        """Unlock store so that other processes can access it."""
        if self._lock is not None:
            self._lock.release()
            log.debug("Store unlocked")

    def _setup_new_zarr_store(
        self,
        parameter_names: ParameterNamesType,
        sim_shapes: ObsShapeType,
        root: zarr.Group,
        chunksize: int = 1,
        sim_dtype: str = "f8",
    ) -> None:  # Adding observational shapes to store
        # Parameters
        n_parameters = len(parameter_names)
        v = root.zeros(
            self._filesystem.v,
            shape=(0, n_parameters),
            chunks=(chunksize, n_parameters),
            dtype="f8",
        )
        v.attrs["parameter_names"] = parameter_names

        # Simulations
        sims = root.create_group(self._filesystem.sims)
        for name, shape in sim_shapes.items():
            sims.zeros(
                name, shape=(0, *shape), chunks=(chunksize, *shape), dtype=sim_dtype
            )

        # Random intensity weights
        root.zeros(self._filesystem.log_w, shape=(0,), chunks=(chunksize,), dtype="f8")

        # Pickled Intensity (prior * N) objects
        root.create(
            self._filesystem.log_lambdas,
            shape=(0,),
            dtype=object,
            object_codec=numcodecs.Pickle(protocol=self._pickle_protocol),
        )

        # Simulation status code
        root.zeros(
            self._filesystem.simulation_status,
            shape=(0,),
            chunks=(chunksize,),
            dtype="int",
        )

        self._update()

    def _update(self) -> None:
        self.sims = self._root[self._filesystem.sims]
        self.v = self._root[self._filesystem.v]
        self.log_w = self._root[self._filesystem.log_w]
        self.log_lambdas = self._root[self._filesystem.log_lambdas]
        self.sim_status = self._root[self._filesystem.simulation_status]
        self.parameter_names = self._root[self._filesystem.v].attrs["parameter_names"]

    def __len__(self) -> int:
        """Returns number of samples in the store."""
        self._update()
        return len(self.v)

    def __getitem__(self, i: int) -> Tuple[Mapping[str, np.ndarray], np.ndarray]:
        """Returns data store entry with index :math:`i`."""
        self._update()
        sim = {}
        for key, value in self.sims.items():
            sim[key] = value[i]
        par = self.v[i]
        return (sim, par)

    def _append_new_points(self, v: Array, log_w: Array) -> None:
        """Append z to zarr_store content and generate new slots for x."""
        self._update()
        n = len(v)
        for key, value in self.sims.items():
            shape = list(value.shape)
            shape[0] += n
            value.resize(*shape)

        self._root[self._filesystem.v].append(v)
        self._root[self._filesystem.log_w].append(log_w)
        m = np.full(n, SimulationStatus.PENDING, dtype="int")
        self.sim_status.append(m)

    def log_lambda(self, z: np.ndarray) -> np.ndarray:
        """Intensity function of the store.

        Args:
            z: Array with the sample parameters. Should have shape (num. samples,
                num. parameters per sample).

        Returns:
            Array with the sample intensities.
        """
        self._update()
        d = -np.inf * np.ones_like(z[:, 0])
        if len(self.log_lambdas) == 0:
            return d
        for i in range(len(self.log_lambdas)):
            pdf = swyft.PriorTruncator.from_state_dict(self.log_lambdas[i]["pdf"])
            N = self.log_lambdas[i]["N"]
            r = pdf.log_prob(z) + np.log(N)
            d = np.where(r > d, r, d)
        return d

    def coverage(
        self, N: int, prior: "swyft.Prior", bound: Optional["swyft.Bound"] = None
    ) -> float:
        """Returns fraction of already stored data points.

        Args:
            N: Number of samples
            prior: Prior
            bound: Bound object for prior truncation

        Returns:
            Fraction of samples that is already covered by content of the store.

        .. note::
            A coverage of zero means that all points need to be newly
            simulated. A coverage of 1.0 means that all points are already
            available for this (truncated) prior.

        .. warning::
            Results are Monte Carlo estimated and subject to sampling noise.
        """
        pdf = swyft.PriorTruncator(prior, bound)
        Nsamples = max(N, 1000)  # At least 1000 test samples
        self._update()

        # Generate new points
        v_prop = pdf.sample(np.random.poisson(Nsamples))
        log_lambda_target = pdf.log_prob(v_prop) + np.log(N)
        log_lambda_store = self.log_lambda(v_prop)
        frac = np.where(
            log_lambda_target > log_lambda_store,
            np.exp(-log_lambda_target + log_lambda_store),
            1.0,
        ).mean()
        return frac

    def sample(
        self,
        N: int,
        prior: "swyft.Prior",
        bound: Optional["swyft.Bound"] = None,
        check_coverage: bool = True,
        add: bool = False,
    ) -> np.ndarray:
        """Return samples from store.

        Args:
            N: Number of samples
            prior: Prior
            bound: Bound object for prior truncation
            check_coverage: Check whether requested points are contained in the store.
            add: If necessary, add requested points to the store.

        Returns:
            Indices: Index list pointing to the relevant store entries.
        """
        if add:
            if self.coverage(N, prior, bound=bound) < 1:
                self.add(N, prior, bound=bound)
        if check_coverage:
            if self.coverage(N, prior, bound=bound) < 1.0:
                raise RuntimeError(
                    "Store does not contain enough samples for your requested intensity function `N * prior`."
                )
        pdf = swyft.PriorTruncator(prior, bound)

        self._update()

        # Select points from cache
        v_store = self.v[:]
        log_w_store = self.log_w[:]
        log_lambda_target = pdf.log_prob(v_store) + np.log(N)
        accept_stored = log_w_store <= log_lambda_target
        indices = np.array(range(len(accept_stored)))[accept_stored]

        return indices

    def _get_indices_to_simulate(
        self, indices: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """
        Determine which samples need to be simulated.

        Args:
            indices: array with the indices of the samples to
            consider. If None, consider all samples.

        Returns:
            array with the sample indices
        """
        status = self.get_simulation_status(indices)
        require_simulation = status == SimulationStatus.PENDING
        idx = np.flatnonzero(require_simulation)
        return indices[idx] if indices is not None else idx

    def _set_simulation_status(
        self, indices: Sequence[int], status: SimulationStatus
    ) -> None:
        """
        Flag the specified samples with the simulation status.

        Args:
            indices: array with the indices of the samples to flag
            status: new status for the samples
        """
        assert status in list(SimulationStatus), f"Unknown status {status}"
        current_status = self.sim_status.oindex[indices]
        if np.any(current_status == status):
            log.warning(
                f"Changing simulation status to {status}, but some simulations have already status {status}"
            )
        self.sim_status.oindex[indices] = status

    def get_simulation_status(
        self, indices: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """Determine the status of sample simulations.

        Args:
            indices: List of indices. If None, check the status of all
                samples

        Returns:
            list of simulation statuses
        """
        self._update()
        return (
            self.sim_status.oindex[indices]
            if indices is not None
            else self.sim_status[:]
        )

    def requires_sim(self, indices: Optional[Sequence[int]] = None) -> bool:
        """Check whether there are parameters which require simulation.

        Args:
            indices: List of indices. If None, check all samples.

        Returns:
            True if one or more samples require simulations, False otherwise.
        """
        self._update()
        return self._get_indices_to_simulate(indices).size > 0

    def _get_indices_failed_simulations(self) -> np.ndarray:
        self._update()
        return np.flatnonzero(self.sim_status == SimulationStatus.FAILED)

    @property
    def any_failed(self) -> bool:
        """Check whether there are parameters which currently lead to a failed simulation."""
        self._update()
        return self._get_indices_failed_simulations().size > 0

    def _add_sim(self, i: int, x: Mapping[str, Array]) -> None:
        for k, v in x.items():
            self.sims[k][i] = v
        self._set_simulation_status(i, SimulationStatus.FINISHED)

    def _failed_sim(self, i: int) -> None:
        self._update()
        self._set_simulation_status(i, SimulationStatus.FAILED)

    def set_simulator(self, simulator: "swyft.Simulator") -> None:
        """(Re)set simulator.

        Args:
            simulator: Simulator.
        """
        if self._simulator is not None:
            log.warning("Simulator already set!  Overwriting.")
        self._simulator = simulator

    def simulate(
        self,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        wait_for_results: Optional[bool] = True,
    ) -> None:
        """Run simulator on parameter store with missing corresponding simulations.

        Args:
            indices: list of sample indices for which a simulation is required
            batch_size: simulations will be submitted in batches of the specified size
            wait_for_results: if True, return only when all simulations are done
        """
        if self._simulator is None:
            log.warning("No simulator specified.  No simulations will run.")
            return

        self.lock()
        self._update()
        idx = self._get_indices_to_simulate(indices)
        self._set_simulation_status(idx, SimulationStatus.RUNNING)
        self.unlock()

        # Run simulations and collect status
        if len(idx) == 0:
            log.debug("No simulations required.")
        else:
            # For the MemoryStore, we need to collect results in memory
            collect_in_memory = (
                True if isinstance(self._zarr_store, zarr.MemoryStore) else False
            )

            if collect_in_memory and not wait_for_results:
                logging.warning(
                    "Asynchronous collection of results is not implemented with the MemoryStore"
                )

            self._simulator._run(
                v=self.v,
                sims={k: v.oindex for k, v in self.sims.items()},
                sim_status=self.sim_status.oindex,
                indices=idx,
                collect_in_memory=collect_in_memory,
                batch_size=batch_size,
            )

        if wait_for_results:
            self.wait_for_simulations(indices)

    def wait_for_simulations(self, indices: Sequence[int]) -> None:
        """Wait for a set of sample simulations to be finished.

        Args:
            indices: list of sample indices
        """
        done = False
        while not done:
            time.sleep(1)
            status = self.get_simulation_status(indices)
            done = np.isin(status, [SimulationStatus.FINISHED, SimulationStatus.FAILED])
            done = np.all(done)

    @classmethod
    def directory_store(
        cls,
        path: PathType,
        simulator: Optional[Simulator] = None,
        sync_path: Optional[PathType] = None,
        overwrite: bool = False,
    ) -> "Store":
        """Instantiate a new Store based on a Zarr DirectoryStore.

        Args:
            path: path to storage directory
            simulator: simulator object
            sync_path: path for synchronization via file locks (files will be stored in the given path).
                It must differ from path, it must be accessible to all processes working on the store,
                and the underlying filesystem must support file locking.
            overwrite: if True, and a store already exists at the specified path, overwrite it.

        Returns:
            Store based on a Zarr DirectoryStore

        Example:
            >>> store = swyft.Store.directory_store(PATH_TO_STORE)
        """
        if not Path(path).exists() or overwrite:
            zarr_store = zarr.DirectoryStore(path)
            sync_path = sync_path or os.path.splitext(path)[0] + ".sync"
            return cls(
                zarr_store=zarr_store,
                simulator=simulator,
                sync_path=sync_path,
                from_scratch=True,
            )
        else:
            raise FileExistsError(
                f"Path {path} exists - set overwrite=True to initialize a new store there."
            )

    @classmethod
    def memory_store(cls, simulator: Simulator) -> "Store":
        """Instantiate a new Store based on a Zarr MemoryStore.

        Args:
            simulator: simulator object

        Returns:
            Store based on a Zarr MemoryStore

        Note:
            The store returned is in general expected to be faster than an equivalent
            store based on the Zarr DirectoryStore, and thus useful for quick
            explorations, or for loading data into memory before training.

        Example:
            >>> store = swyft.Store.memory_store(simulator)
        """
        zarr_store = zarr.MemoryStore()
        return cls(zarr_store=zarr_store, simulator=simulator, from_scratch=True)

    def save(self, path: PathType) -> None:
        """Save the Store to disk using a Zarr DirectoryStore.

        Args:
            path: path where to create the Zarr root directory
        """
        if isinstance(
            self._zarr_store, zarr.DirectoryStore
        ) and self._zarr_store.path == os.path.abspath(path):
            return
        path = Path(path)
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"{path} should be a directory")
        elif path.exists() and not is_empty(path):
            raise FileExistsError(f"{path} is not empty")
        else:
            path.mkdir(parents=True, exist_ok=True)
            zarr_store = zarr.DirectoryStore(path)
            zarr.convenience.copy_store(source=self._zarr_store, dest=zarr_store)

    @classmethod
    def load(
        cls,
        path: PathType,
        simulator: Optional[Simulator] = None,
        sync_path: Optional[PathType] = None,
    ) -> "Store":
        """Open an existing sample store using a Zarr DirectoryStore.

        Args:
            path: path to the Zarr root directory
            simulator: simulator object
            sync_path: path for synchronization via file locks (files will be stored in the given path).
                It must differ from path, it must be accessible to all processes working on the store,
                and the underlying filesystem must support file locking.
        """
        if Path(path).exists():
            store = zarr.DirectoryStore(path)
            sync_path = sync_path or os.path.splitext(path)[0] + ".sync"
            return cls(
                zarr_store=store,
                simulator=simulator,
                sync_path=sync_path,
                from_scratch=False,
            )
        else:
            raise FileNotFoundError(f"There is no directory store at {path}.")

    def to_memory(self) -> "Store":
        """Make an in-memory copy of the existing Store using a Zarr MemoryStore."""
        memory_store = zarr.MemoryStore()
        zarr.convenience.copy_store(source=self._zarr_store, dest=memory_store)
        return Store(
            zarr_store=memory_store, simulator=self._simulator, from_scratch=False
        )
