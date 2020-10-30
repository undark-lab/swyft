# pylint: disable=no-member, not-callable
import torch
import numpy as np

import zarr
import numcodecs

from tqdm import tqdm


class DataStore:
    """The iP3 datastore.

    Args:
        filename (str): Optional. If given, defines path of `zarr.DirectoryStore`.
    """

    def __init__(self, filename=None):
        # Open (new) datastore
        if filename is None:
            self.store = zarr.MemoryStore()
        else:
            self.store = zarr.DirectoryStore(filename)
        self.root = zarr.group(store=self.store)

        if "samples" not in self.root.keys():
            print("Creating empty datastore:", filename)
            print("...don't forget to run `init` to set up storage parameters.")
            return

        print("Loading datastore:", filename)
        self._update()

    def _update(self):
        self.x = self.root["samples/x"]
        self.z = self.root["samples/z"]
        self.m = self.root["metadata/needs_sim"]
        self.u = self.root["metadata/intensity"]

    def init(self, xdim, zdim):
        """Initialize data store content dimensions.

        Args:
            zdim (int): Number of z dimensions
            xdim (tuple): Shape of x array
        """
        if "samples" in self.root.keys():
            print("WARNING: Datastore is already initialized.")
            return self
        self.x = self.root.zeros(
            "samples/x", shape=(0,) + xdim, chunks=(1,) + xdim, dtype="f4"
        )
        self.z = self.root.zeros(
            "samples/z", shape=(0,) + (zdim,), chunks=(10000,) + (zdim,), dtype="f4"
        )
        self.m = self.root.zeros(
            "metadata/needs_sim", shape=(0, 1), chunks=(10000,) + (1,), dtype="bool"
        )
        self.u = self.root.create(
            "metadata/intensity",
            shape=(0,),
            dtype=object,
            object_codec=numcodecs.Pickle(),
        )
        print("Datastore initialized.")

        return self

    def _append_z(self, z):
        """Append z to datastore content and new slots for x."""
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

    def __len__(self):
        """Returns number of samples in the datastore."""
        self._update()

        return len(self.z)

    def intensity(self, zlist):
        """Evaluate DataStore intensity function.

        Args:
            z (array-like): list of parameter values.
        """
        self._update()

        if len(self.u) == 0:
            return np.zeros(len(zlist))
        else:
            return np.array([self.u[i](zlist) for i in range(len(self.u))]).max(axis=0)

    def _grow(self, p):
        """Grow number of samples in datastore."""
        # Proposed new samples z from p
        z_prop = p.sample()

        # Rejection sampling from proposal list
        accepted = []
        ds_intensities = self.intensity(z_prop)
        target_intensities = p(z_prop)
        for z, Ids, It in zip(z_prop, ds_intensities, target_intensities):
            rej_prob = np.minimum(1, Ids / It)
            w = np.random.rand(1)[0]
            accepted.append(rej_prob < w)
        z_accepted = z_prop[accepted, :]

        # Add new entries to datastore and update intensity function
        self._append_z(z_accepted)
        if len(z_accepted) > 0:
            self.u.resize(len(self.u) + 1)
            self.u[-1] = p
            print("Adding %i new samples. Run simulator!" % len(z_accepted))
        else:
            print("No new simulator runs required.")

    def sample(self, p):
        """Sample from DataStore.

        Args:
            p (intensity function): Target intensity function.
        """
        self._update()

        self._grow(p)

        accepted = []
        zlist = self.z[:]
        I_ds = self.intensity(zlist)
        I_target = p(zlist)
        for i, z in enumerate(zlist):
            accept_prob = I_target[i] / I_ds[i]
            assert (
                accept_prob <= 1.0
            ), "Inconsistent intensity function of data store. This should not happen."
            w = np.random.rand(1)[0]
            if accept_prob > w:
                accepted.append(i)
        return accepted

    def __getitem__(self, i):
        self._update()

        return self.x[i], self.z[i]

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


class Mask1d:
    """A 1-dim multi-interval based mask class."""

    def __init__(self, intervals):
        self.intervals = np.array(intervals)  # n x 2 matrix

    def __call__(self, z):
        """Returns 1. if inside interval, otherwise 0."""
        m = np.zeros_like(z)
        for z0, z1 in self.intervals:
            m += np.where(z >= z0, np.where(z <= z1, 1.0, 0.0), 0.0)
        assert not any(m > 1.0), "Overlapping intervals."
        return m

    def area(self):
        """Combined length of all intervals (AKAK 1-dim area)."""
        return (self.intervals[:, 1] - self.intervals[:, 0]).sum()

    def sample(self, N):
        p = self.intervals[:, 1] - self.intervals[:, 0]
        p /= p.sum()
        i = np.random.choice(len(p), size=N, replace=True, p=p)
        w = np.random.rand(N)
        z = self.intervals[i, 0] + w * (self.intervals[i, 1] - self.intervals[i, 0])
        return z


class FactorMask:
    """A d-dim factorized mask."""

    def __init__(self, masks):
        self.masks = masks
        self.d = len(masks)

    def __call__(self, z):
        m = [self.masks[i](z[:, i]) for i in range(self.d)]
        m = np.array(m).prod(axis=0)
        return m

    def area(self):
        m = [self.masks[i].area() for i in range(self.d)]
        return np.array(m).prod()

    def sample(self, N):
        z = np.empty((N, self.d))
        for i in range(self.d):
            z[:, i] = self.masks[i].sample(N)
        return z


class Intensity:
    """Intensity function based on d-dim mask."""

    def __init__(self, mu, mask):
        self.mu = mu
        self.mask = mask
        self.area = mask.area()

    def __call__(self, z):
        return self.mask(z) / self.area * self.mu

    def sample(self, N=None):
        if N is None:
            N = np.random.poisson(self.mu, 1)[0]
        return self.mask.sample(N)


def construct_intervals(x, y):
    """Get x intervals where y is above 0."""
    indices = np.argsort(x)
    x = x[indices]
    y = y[indices]
    m = np.where(y > 0.0, 1.0, 0.0)
    m = m[1:] - m[:-1]
    i0 = np.argwhere(m == 1.0)[:, 0]  # Upcrossings
    i1 = np.argwhere(m == -1.0)[:, 0]  # Downcrossings

    # No crossings --> return entire interval
    if len(i0) == 0 and len(i1) == 0:
        return [[x[0], x[-1]]]

    # One more upcrossing than downcrossing
    # --> Treat right end as downcrossing
    if len(i0) - len(i1) == 1:
        i1 = np.append(i1, -1)

    # One more downcrossing than upcrossing
    # --> Treat left end as upcrossing
    if len(i0) - len(i1) == -1:
        i0 = np.append(0, i0)

    intervals = []
    for i in range(len(i0)):
        intervals.append([x[i0[i]], x[i1[i]]])

    return intervals


class DataContainer(torch.utils.data.Dataset):
    """Simple data container class.

    Note: The noisemodel allows scheduled noise level increase during training.
    """

    def __init__(self, datastore, indices, noisemodel=None):
        super().__init__()
        # Check whether datastore is complete
        if datastore.requires_sim():
            raise RuntimeError("Datastore entries missing. Run simulator.")

        self.ds = datastore
        self.indices = indices
        self.noisemodel = noisemodel

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Obtain x, z
        i = self.indices[idx]
        x = self.ds.x[i]
        z = self.ds.z[i]

        # Add optional noise
        if self.noisemodel is not None:
            x = self.noisemodel(x, z)

        # Tensors
        x = torch.tensor(x).float()
        z = torch.tensor(z).float()

        # Done
        xz = dict(x=x, z=z)
        return xz


if __name__ == "__main__":
    pass
