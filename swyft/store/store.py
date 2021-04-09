# pylint: disable=no-member, not-callable
import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Callable, Dict, List, Union
from warnings import warn

import numcodecs
import numpy as np
import zarr
from tqdm import tqdm

from swyft.store.exceptions import LowIntensityError
from swyft.types import Array, PathType, Shape
from swyft.utils import all_finite, is_empty
from swyft.marginals.prior import BoundedPrior

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


class Store(ABC):
    """Abstract base class for various stores."""

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
        simulator = None,
    ):
        """Initialize Store content dimensions.

        Args:
            params (list of strings or int): List of paramater names.  If int use ['z0', 'z1', ...].
            obs_shapes (dict): Map of obs names to shapes
            store: zarr storage.
        """
        if isinstance(params, int):
            params = ['z%i'%i for i in range(params)]

        self.store = store
        self.params = params
        self.root = zarr.group(store=self.store)
        #self.intensities = []

        self._simulator = simulator

        logging.debug("Creating Store.")
        logging.debug("  params = %s" % str(params))
        logging.debug("  obs_shapes = %s" % str(obs_shapes))

        if all(key in self.root.keys() for key in ["samples", "metadata"]):
            logging.info("Loading existing store.")
            self._update()
        elif len(self.root.keys()) == 0:
            logging.info("Creating new store.")
            self._setup_new_store(len(params), obs_shapes, self.root)
        else:
            raise KeyError(
                "The zarr storage is corrupted. It should either be empty or only have the keys ['samples', 'metadata']."
            )

    def _setup_new_store(self, zdim, obs_shapes, root) -> None: # Adding observational shapes to store
        x = root.create_group(self._filesystem.obs)
        for name, shape in obs_shapes.items():
            x.zeros(name, shape=(0, *shape), chunks=(1, *shape), dtype="f8")

        z = root.zeros(  # noqa: F841
            "z",
            shape=(0, zdim),
            chunks=(100000, 1),
            dtype="f8",
        )

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
        return group[Store._filesystem.obs].shape[1:]

    @staticmethod
    def _extract_zdim_from_zarr_group(group):
        return group[Store._filesystem.par].shape[1]

    # FIXME: Add log_intensity as another entry to datastore
    def _update(self):
        # This could be removed with a property for each attribute which only loads from disk if something has changed. TODO
        # FIXME: Update BallTree necessary for intensity calculations
        # - Distances should be calculated based on prior hypercube projection
        # - This is only clear when running grow() or sample()
        self.x = self.root[self._filesystem.obs]
        #self.z = self.root[self._filesystem.par]
        self.z = self.root['z']
        self.m = self.root[self._filesystem.requires_simulation]
        self.f = self.root[self._filesystem.failed_simulation]
        assert len(self.f) == len(
            self.m
        ), "Metadata noting which indices require simulation and which have failed have desynced."
        self.u = self.root[self._filesystem.intensity]
        self.wu = self.root[self._filesystem.which_intensity]

    def __len__(self):
        """Returns number of samples in the store."""
        self._update()
        return len(self.z)

    def __getitem__(self, i):
        self._update()

        result_x = {}
        for key, value in self.x.items():
            result_x[key] = value[i]

        result_z = self.z[i]

        return dict(x=result_x, z=result_z)

    def _append_z(self, z):
        """Append z to store content and generate new slots for x."""
        self._update()

        # Length of first element
        n = len(z)
        #n = len(z[list(z)[0]])

        # Add slots for x
        for key, value in self.x.items():
            shape = list(value.shape)
            shape[0] += n
            value.resize(*shape)

        self.root['z'].append(z)
        #for key, value in self.z.items():
        #    value.append(z[key])

        # Register as missing
        m = np.ones((n, 1), dtype="bool")
        self.m.append(m)

        # Simulations have not failed, yet.
        self.f.append(~m)

        ## Which intensity was a parameter drawn with
        #wu = self.intensity_len * np.ones(n, dtype="int")
        #self.wu.append(wu)

    def log_intensity(self, z: np.ndarray) -> np.ndarray:
        self._update()
        d = -np.inf * np.ones_like(z[:,0])
        if len(self.u) == 0:
            return d
        for i in range(len(self.u)):
            pdf = BoundedPrior.from_state_dict(self.u[i]['pdf'])
            N = self.u[i]['N']
            r = pdf.log_prob(z) + np.log(N)
            d = np.where(r > d, r, d)
        return d

    def grow(self, N, pdf):  # noqa
        """Given an intensity function, add parameter samples to the store.

        Args:
            intensity: target parameter intensity function
        """
        # Random Poisson realization - relevant for PPP logic
        z_prop = pdf.sample(N = np.random.poisson(N))

        # Rejection sampling from proposal list
        accepted = []

        target_log_intensities = pdf.log_prob(z_prop) + np.log(N)
        stored_log_intensities = self.log_intensity(z_prop)

        for stored_log_intensity, target_log_intensity in zip(
            stored_log_intensities, target_log_intensities
        ):
            log_prob_reject = np.minimum(0, stored_log_intensity - target_log_intensity)
            accepted.append(log_prob_reject < np.log(np.random.rand()))
        z_accepted = z_prop[accepted]

        if sum(accepted) > 0:
            # Add new entries to store
            self._append_z(z_accepted)
            logging.info("  adding %i new samples to simulator store." % sum(accepted))

        # And update intensity function
        self.u.resize(len(self.u) + 1)
        self.u[-1] = dict(pdf = pdf.state_dict(), N = N)

    # FIXME: No Balltree required here, just rejection sampling based on stored intensities.
    def sample(self, N: int, prior) -> List[int]:  # noqa: F821
        self._update()

        accepted = []
        #zlist = {k: self.z[k][:] for k in (self.z)}
        zlist = self.z

        stored_intensities = self.log_intensity(zlist)
        target_intensities = prior.log_prob(zlist) + np.log(N)
        assert len(self) == len(stored_intensities)
        assert len(self) == len(target_intensities)
        for i, (target_intensity, stored_intensity) in enumerate(
            zip(target_intensities, stored_intensities)
        ):
            log_prob_accept = target_intensity - stored_intensity
            if log_prob_accept > 0.0:
                print(target_intensity)
                print(stored_intensity)
                raise LowIntensityError(
                    f"{log_prob_accept} > 0, "
                    " but we expected the log ratio of target intensity function to the store <= 0. "
                    "There may not be enough samples in the store or "
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

    def set_simulator(self, simulator):
        self._simulator = simulator

    def simulate(
        self,
        fail_on_non_finite: bool = True,
        max_attempts: int = 1000,
    ) -> None:
        """Run simulator sequentially on parameter store with missing corresponding simulations.

        Args:
            simulator: simulates an observation given a parameter input
            fail_on_non_finite: if nan / inf in simulation, considered a failed simulation
            max_attempts: maximum number of resample attempts before giving up.
        """
        self._update()

        if self._simulator is None:
            logging.warning("No simulator specified")
            return
        else:
            simulator = self._simulator

        idx = self._get_idx_requiring_sim()
        if len(idx) == 0:
            logging.debug("No simulations required.")
            return True
        for i in tqdm(idx, desc="Simulate"):
            #z = {k: v[i] for k, v in self.z.items()}
            z = self.z[i]
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
                f"Some simulations failed, despite {max_attempts} to resample them. They have been marked in the store."
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
            logging.debug("No failed simulations.")
            return None


class DirectoryStore(Store):
    def __init__(self, params, obs_shapes: Shape, path: PathType):
        """Instantiate an iP3 store stored in a directory.

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
        return DirectoryStore(zdim=zdim, xshape=xshape, path=path)


class MemoryStore(Store):
    def __init__(self, params, obs_shapes, store=None, simulator=None):
        """Instantiate an iP3 store stored in the memory.

        Args:
            zdim: Number of z dimensions
            obs_shapes: Shape of x array
            store (zarr.MemoryStore, zarr.DirectoryStore): optional, used in loading.
        """
        if store is None:
            self.store = zarr.MemoryStore()
            logging.debug("Creating new empty MemoryStore.")
        else:
            self.store = store
            logging.debug("Creating MemoryStore from store.")
        super().__init__(params=params, obs_shapes=obs_shapes, store=self.store, simulator=simulator)

    def save(self, path: PathType) -> None:
        """Save the current state of the MemoryStore to a directory."""
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
        """Load existing DirectoryStore state into a MemoryStore object."""
        memory_store = zarr.MemoryStore()
        directory_store = zarr.DirectoryStore(path)
        zarr.convenience.copy_store(source=directory_store, dest=memory_store)

        group = zarr.group(store=memory_store)
        xshape = cls._extract_xshape_from_zarr_group(group)
        zdim = cls._extract_zdim_from_zarr_group(group)
        return MemoryStore(zdim=zdim, xshape=xshape, store=memory_store)

    @classmethod
    def from_simulator(cls, model, prior, noise = None):
        """Convenience function to instantiate new MemoryStore with correct obs_shapes.

        Args:
            model (function): Simulator model.
            prior (Prior): Model prior.

        Note:
            The simulator model is run once in order to infer observable shapes from the output.
        """
        v = prior.sample(1)[0]
        vdim = len(v)

        obs = model(v)
        if noise is not None:
            obs = noise(obs, v)

        obs_shapes = {k: v.shape for k, v in obs.items()}

        return MemoryStore(vdim, obs_shapes)
