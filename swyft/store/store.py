# pylint: disable=no-member, not-callable
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import fasteners
import numcodecs
import numpy as np
import zarr

import swyft

log = logging.getLogger(__name__)

from swyft.store.simulator import SimulationStatus, Simulator
from swyft.types import PathType
from swyft.utils import is_empty


class Filesystem:
    metadata = "metadata"
    log_lambdas = "metadata/log_lambdas"
    samples = "samples"
    sims = "samples/sims"
    v = "samples/v"
    log_w = "samples/log_w"
    simulation_status = "samples/simulation_status"


class Store(ABC):
    """Abstract base class for various stores.

    Args:
        zarr_store: zarr storage.
        simulator: simulator object.
        sync_path: if specified, it will enable synchronization using file locks (files will be
            stored in the given path). Must be accessible to all processes working on the store
            and the underlying filesystem must support file locking.
        chunksize: the parameters and simulation output will be stored as arrays with the
            specified chunk size along the sample dimension (a single chunk will be used for the
            other dimensions).
        pickle_protocol: pickle protocol number used for storing intensity functions.
    """

    _filesystem = Filesystem

    @abstractmethod
    def __init__(
        self,
        zarr_store: Union[zarr.MemoryStore, zarr.DirectoryStore],
        simulator: Optional[Simulator] = None,
        sync_path: Optional[PathType] = None,
        chunksize: int = 1,
        pickle_protocol: int = 4,
    ) -> None:
        self._zarr_store = zarr_store
        self._simulator = simulator
        self._pickle_protocol = pickle_protocol  # TODO: to be deprecated, we will default to 4, which is supported since python 3.4

        synchronizer = zarr.ProcessSynchronizer(sync_path) if sync_path else None
        self._root = zarr.group(store=self._zarr_store, synchronizer=synchronizer)

        if set(["samples", "metadata"]) == set(self._root.keys()):
            print("Loading existing store.")
            self._update()
        elif len(self._root.keys()) == 0:
            print("Creating new store.")

            # TODO: Remove
            #            log.debug("Loading existing store.")
            #            self._update()
            #        elif len(self._root.keys()) == 0:
            #            log.debug("Creating new store.")

            self._setup_new_zarr_store(
                simulator.pnames, simulator.sim_shapes, self._root, chunksize=chunksize
            )
            log.debug("  sim_shapes = %s" % str(simulator.sim_shapes))
        else:
            raise KeyError(
                "The zarr storage is corrupted. It should either be empty or only have the keys ['samples', 'metadata']."
            )

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
        pdf = swyft.TruncatedPrior(prior, bound)

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

    def _setup_lock(self, sync_path):
        path = os.path.join(sync_path, "cache.lock")
        self._lock = fasteners.InterProcessLock(path)

    def lock(self) -> None:
        if self._lock is not None:
            log.debug("Cache locked")
            self._lock.acquire(blocking=True)

    def unlock(self) -> None:
        if self._lock is not None:
            self._lock.release()
            log.debug("Cache unlocked")

    def _setup_new_zarr_store(
        self, pnames, sim_shapes, root, chunksize=1
    ) -> None:  # Adding observational shapes to store
        # Parameters
        zdim = len(pnames)
        v = root.zeros(
            self._filesystem.v, shape=(0, zdim), chunks=(chunksize, zdim), dtype="f8"
        )
        v.attrs["pnames"] = pnames

        # Simulations
        sims = root.create_group(self._filesystem.sims)
        for name, shape in sim_shapes.items():
            sims.zeros(name, shape=(0, *shape), chunks=(chunksize, *shape), dtype="f8")

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

    def _update(self) -> None:
        self.sims = self._root[self._filesystem.sims]
        self.v = self._root[self._filesystem.v]
        self.log_w = self._root[self._filesystem.log_w]
        self.log_lambdas = self._root[self._filesystem.log_lambdas]
        self.sim_status = self._root[self._filesystem.simulation_status]
        self.pnames = self._root[self._filesystem.v].attrs["pnames"]

    def __len__(self):
        """Returns number of samples in the store."""
        self._update()
        return len(self.v)

    def __getitem__(self, i):
        """Returns data store entry with index :math:`i`."""
        self._update()
        sim = {}
        for key, value in self.sims.items():
            sim[key] = value[i]
        par = self.v[i]
        return (sim, par)

    def _append_new_points(self, v, log_w):
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
        """Intensity function of the store."""
        self._update()
        d = -np.inf * np.ones_like(z[:, 0])
        if len(self.log_lambdas) == 0:
            return d
        for i in range(len(self.log_lambdas)):
            pdf = swyft.TruncatedPrior.from_state_dict(self.log_lambdas[i]["pdf"])
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
        pdf = swyft.TruncatedPrior(prior, bound)
        Nsamples = max(N, 1000)  # At least 1000 test samples
        self._update()

        # Generate new points
        z_prop = pdf.sample(N=np.random.poisson(Nsamples))
        log_lambda_target = pdf.log_prob(z_prop) + np.log(N)
        log_lambda_store = self.log_lambda(z_prop)
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
            N (int): Number of samples
            prior (swyft.Prior): Prior
            bound (swyft.Bound): Bound object for prior truncation
            check_coverage (bool): Check whether requested points are contained in the store.
            add (bool): If necessary, add requested points to the store.

        Returns:
            Indices (list): Index list pointing to the relevant store entries.
        """
        if add:
            if self.coverage(N, prior, bound=bound) < 1:
                self.add(N, prior, bound=bound)
        if check_coverage:
            if self.coverage(N, prior, bound=bound) < 1.0:
                print("WARNING: Store does not contain enough samples.")
                return []
        pdf = swyft.TruncatedPrior(prior, bound)

        self._update()

        # Select points from cache
        z_store = self.v[:]
        log_w_store = self.log_w[:]
        log_lambda_target = pdf.log_prob(z_store) + np.log(N)
        accept_stored = log_w_store <= log_lambda_target
        indices = np.array(range(len(accept_stored)))[accept_stored]

        return indices

    def _get_indices_to_simulate(self, indices: Optional[Sequence[int]] = None):
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

    def _set_simulation_status(self, indices: Sequence[int], status: SimulationStatus):
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

    def get_simulation_status(self, indices: Optional[Sequence[int]] = None):
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
        """Check whether there are parameters which require simulation."""
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

    # NOTE: Deprecated
    #    @staticmethod
    #    def did_simulator_succeed(x: Dict[str, Array], fail_on_non_finite: bool) -> bool:
    #        """Is the simulation a success?"""
    #
    #        assert isinstance(x, dict), "Simulators must return a dictionary."
    #
    #        dict_anynone = lambda d: any(v is None for v in d.values())  # noqa: E731
    #
    #        if dict_anynone(x):
    #            return False
    #        elif fail_on_non_finite and not all_finite(x):
    #            return False
    #        else:
    #            return True

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
        """Run simulator sequentially on parameter store with missing corresponding simulations.

        Args:
            indices (list): list of sample indices for which a simulation is required
            batch_size (int): simulations will be submitted in batches of the specified
                size
            wait_for_results (bool): if True, return only when all simulations are done
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
            return
        else:
            # For the MemoryStore, we need to collect results in memory
            collect_in_memory = True if isinstance(self, MemoryStore) else False

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

    def wait_for_simulations(self, indices: Sequence[int]):
        """Wait for a set of sample simulations to be finished.

        Args:
            indices (list): list of sample indices
        """
        done = False
        while not done:
            time.sleep(1)
            status = self.get_simulation_status(indices)
            done = np.isin(status, [SimulationStatus.FINISHED, SimulationStatus.FAILED])
            done = np.all(done)


class DirectoryStore(Store):
    """Instantiate DirectoryStore.

    Args:
        path (PathType): path to storage directory
        simulator (swyft.Simulator): simulator object
        sync_path: path for synchronization via file locks (files will be stored in the given path).
            It must differ from path, it must be accessible to all processes working on the store,
            and the underlying filesystem must support file locking.

    Example::

        >>> store = swyft.DirectoryStore(PATH_TO_STORE)
        >>> print("Number of simulations in store:", len(store))
    """

    def __init__(
        self,
        path: PathType,
        simulator: Optional[Simulator] = None,
        sync_path: Optional[PathType] = None,
    ) -> None:
        zarr_store = zarr.DirectoryStore(path)
        sync_path = sync_path or os.path.splitext(path)[0] + ".sync"
        super().__init__(
            zarr_store=zarr_store, simulator=simulator, sync_path=sync_path
        )


class MemoryStore(Store):
    """Instantiate a new memory store for a given simulator.

    Args:
        simulator (swyft.Simulator): Simulator object

    .. note::
        The swyft.MemoryStore is in general expected to be faster than
        swyft.DirectoryStore, and useful for quick explorations, or for
        loading training data into memory before training.

    Example::

        >>> store = swyft.MemoryStore(simulator)
    """

    def __init__(self, simulator: Simulator) -> None:
        zarr_store = zarr.MemoryStore()
        super().__init__(zarr_store=zarr_store, simulator=simulator)

    def save(self, path: PathType) -> None:
        """Save the MemoryStore to a DirectoryStore.

        Args:
            path (PathType): Path to DirectoryStore.
        """
        path = Path(path)
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"{path} should be a directory")
        elif path.exists() and not is_empty(path):
            raise FileExistsError(f"{path} is not empty")
        else:
            path.mkdir(parents=True, exist_ok=True)
            zarr_store = zarr.DirectoryStore(path)
            zarr.convenience.copy_store(source=self._zarr_store, dest=zarr_store)
            return None

    @classmethod
    def load(cls, path: PathType):
        """Load existing DirectoryStore into a MemoryStore.

        Args:
            path (PathType): path to DirectoryStore
        """
        memory_store = zarr.MemoryStore()
        directory_store = zarr.DirectoryStore(path)
        zarr.convenience.copy_store(source=directory_store, dest=memory_store)
        obj = MemoryStore.__new__(MemoryStore)
        super(MemoryStore, obj).__init__(zarr_store=memory_store, simulator=None)
        return obj


#    def copy(self, sync_path=None):
#        zarr_store = zarr.MemoryStore()
#        zarr.convenience.copy_store(source=self._zarr_store, dest=zarr_store)
#        return MemoryStore(
#            params=self.params,
#            zarr_store=zarr_store,
#            simulator=self._simulator,
#            sync_path=sync_path,
#        )
#            zarr_store (zarr.MemoryStore, zarr.DirectoryStore): optional, used in
#                loading.
