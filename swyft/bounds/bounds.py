import logging

import numpy as np
import torch
from sklearn.neighbors import BallTree

log = logging.getLogger(__name__)


class Bound:
    """A bound region on the hypercube.

    .. note::
        The Bound object provides methods to sample from subregions of the
        hypercube, to evaluate the volume of the constrained region, and to
        evaluate the bound.
    """

    def __init__(self):
        pass

    @property
    def volume(self):
        """Volume of the bound region."""
        raise NotImplementedError

    @property
    def udim(self):
        """Number of dimensions."""
        raise NotImplementedError

    def sample(self, N):
        """Sample.

        Args:
            N (int): Numbe of samples.

        Returns:
            s (N x udim np.ndarray)
        """
        raise NotImplementedError

    def __call__(self, u):
        """Check whether parameters are within bounds.

        Args:
            u (N x udim np.ndarray): Parameters on hypercube
        """
        raise NotImplementedError

    @classmethod
    def from_state_dict(cls, state_dict):
        """Instantiate Bound object based on state_dict.

        Args:
            state_dict (dict): State dictionary
        """
        tag = state_dict["tag"]
        if tag == "UnitCubeBound":
            return UnitCubeBound.from_state_dict(state_dict)
        elif tag == "RectangleBound":
            return RectangleBound.from_state_dict(state_dict)
        elif tag == "BallsBound":
            return BallsBound.from_state_dict(state_dict)
        elif tag == "CompositBound":
            return CompositBound.from_state_dict(state_dict)
        else:
            raise KeyError

    @classmethod
    def load(cls, filename):
        sd = torch.load(filename)
        return cls.from_state_dict(sd)

    def save(self, filename):
        sd = self.state_dict()
        torch.save(sd, filename)

    @classmethod
    def from_Posteriors(cls, partitions, post, obs, th=-13.0):
        return CompositBound.from_Posteriors(partitions, post, obs, post.bound, th=th)


class UnitCubeBound(Bound):
    """The unit hypercube bound."""

    def __init__(self, udim):
        """Initialize unit hypercube bound.

        Args:
            udim (int): Number of parameters.
        """
        self._udim = udim
        self._volume = 1.0

    @property
    def volume(self):
        """The volume of the constrained region."""
        return self._volume

    @property
    def udim(self):
        """Dimensionality of the constrained region."""
        return self._udim

    def sample(self, N):
        """Generate samples from the bound region.

        Args:
            N (int): Number of samples
        """
        return np.random.rand(N, self.udim)

    def __call__(self, u):
        """Evaluate bound.

        Args:
            u (array): Input array.

        Returns:
            Ones and zeros
        """
        b = np.where(u <= 1.0, np.where(u >= 0.0, 1.0, 0.0), 0.0)
        return b.prod(axis=-1)

    def state_dict(self):
        return dict(tag="UnitCubeBound", udim=self.udim)

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(state_dict["udim"])


class RectangleBound(Bound):
    def __init__(self, rec_bounds):
        """Rectangle bound.

        Args:
            rec_bounds (n x 2 np.ndarray): list of (u_min, u_max) values.

        Note: 0 <= u_min < u_max  <= 1.
        """
        self._rec_bounds = rec_bounds

    @property
    def volume(self):
        V = 1.0
        for i in range(self.udim):
            V *= self._rec_bounds[i, 1] - self._rec_bounds[i, 0]
        return V

    @property
    def udim(self):
        return len(self._rec_bounds)

    def sample(self, N):
        u = np.random.rand(N, self.udim)
        for i in range(self.udim):
            u[:, i] *= self._rec_bounds[i, 1] - self._rec_bounds[i, 0]
            u[:, i] += self._rec_bounds[i, 0]
        return u

    def __call__(self, u):
        m = np.ones(len(u))
        for i, v in enumerate(self._rec_bounds):
            m *= np.where(u[:, i] >= v[0], np.where(u[:, i] <= v[1], 1.0, 0.0), 0.0)
        return m > 0.5

    def state_dict(self):
        return dict(tag="RectangleBound", rec_bounds=self._rec_bounds)

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(state_dict["rec_bounds"])


class BallsBound(Bound):
    def __init__(self, points, scale=1.0):
        """Simple mask based on coverage balls around inducing points.

        Args:
            points (Array): shape (num_points, n_dim)
            scale (float): Scale ball size (default 1.0)
        """
        assert len(points.shape) == 2
        self.X = points
        self._udim = self.X.shape[-1]
        self.bt = BallTree(self.X, leaf_size=2)
        self.epsilon = self._set_epsilon(self.X, self.bt, scale)
        self._volume = self._get_volume(self.X, self.epsilon, self.bt)

    @property
    def volume(self):
        return self._volume

    @property
    def udim(self):
        return self._udim

    @staticmethod
    def _set_epsilon(X, bt, scale):
        dims = X.shape[-1]
        k = [4, 5, 6]
        dist, ind = bt.query(X, k=k[dims - 1])  # 4th NN
        epsilon = np.median(dist[:, -1]) * scale * 1.5
        return epsilon

    @staticmethod
    def _get_volume(X, epsilon, bt):
        N = 100
        vol_est = []
        d = X.shape[-1]
        area = {1: 2 * epsilon, 2: np.pi * epsilon ** 2}[d]
        for i in range(N):
            n = np.random.randn(*X.shape)
            norm = (n ** 2).sum(axis=1) ** 0.5
            n = n / norm.reshape(-1, 1)
            r = np.random.rand(len(X)) ** (1 / d) * epsilon
            Y = X + n * r.reshape(-1, 1)
            in_bounds = ((Y >= 0.0) & (Y <= 1.0)).prod(axis=1, dtype="bool")
            Y = Y[in_bounds]
            counts = bt.query_radius(Y, epsilon, count_only=True)
            vol_est.append(area * sum(1.0 / counts))
        vol_est = np.array(vol_est)
        out, err = vol_est.mean(), vol_est.std() / np.sqrt(N)
        rel = err / out
        if rel > 0.01:
            log.debug("WARNING: Rel volume uncertainty is %.4g" % rel)
        return out

    def sample(self, N):
        counter = 0
        samples = []
        d = self.X.shape[-1]
        while counter < N:
            n = np.random.randn(*self.X.shape)
            norm = (n ** 2).sum(axis=1) ** 0.5
            n = n / norm.reshape(-1, 1)
            r = np.random.rand(len(self.X)) ** (1 / d) * self.epsilon
            Y = self.X + n * r.reshape(-1, 1)
            in_bounds = ((Y >= 0.0) & (Y <= 1.0)).prod(axis=1, dtype="bool")
            Y = Y[in_bounds]
            counts = self.bt.query_radius(Y, r=self.epsilon, count_only=True)
            p = 1.0 / counts
            w = np.random.rand(len(p))
            Y = Y[p >= w]
            samples.append(Y)
            counter += len(Y)
        samples = np.vstack(samples)
        ind = np.random.choice(range(len(samples)), size=N, replace=False)
        return samples[ind]

    def __call__(self, u):
        u = u.reshape(len(u), -1)
        dist, ind = self.bt.query(u, k=1)
        return (dist < self.epsilon)[:, 0]

    def state_dict(self):
        return dict(
            tag="BallsBound", points=self.X, epsilon=self.epsilon, volume=self._volume
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        obj = cls.__new__(cls)
        obj.X = state_dict["points"]
        assert len(obj.X.shape) == 2
        obj._udim = obj.X.shape[-1]
        obj.epsilon = state_dict["epsilon"]
        obj._volume = state_dict["volume"]
        obj.bt = BallTree(obj.X, leaf_size=2)
        return obj


class CompositBound(Bound):
    """Composit bound object. Product of multiple bounds."""

    def __init__(self, bounds_map, udim):
        """
        Args:
            bounds_map (dict): Dictionary mapping indices like (0, 3) etc --> bounds
            udim (int): Length of parameter vector.
        """
        self._bounds = bounds_map
        self._udim = udim

    def sample(self, N):
        results = -np.ones((N, self._udim))
        for k, v in self._bounds.items():
            results[:, np.array(k)] = v.sample(N)
        return results

    @property
    def volume(self):
        volume = 1.0
        for k, v in self._bounds.items():
            volume *= v.volume
        return volume

    @property
    def udim(self):
        return self._udim

    def __call__(self, u):
        res = []
        for k, v in self._bounds.items():
            r = v(u[:, np.array(k)])
            res.append(r)
        return sum(res) == len(res)

    # - Function: Generate sample from posterior
    #   - Constraints are based on p(u|z)/p(u), and should be (different from what we have in the paper???)
    #   - That means I need weights without prior corrections, can be an option to switch this on or off
    #   - Samples should be samples from the hypercube
    #   - Use sampled points above a threshold for generating Rec bound and BallBounds, directly based on points
    # - Function: Return isolated ratio function & bound object from Posteriors object
    #   - Can be used in sampling

    @classmethod
    def from_Posteriors(cls, partition, post, obs, bound, th=-13.0, N=10000):
        """Generate new CompositBound object based on RatioEstimator.

        Args:
            rc (RatioEstimator): RatioEstimator to evaluate.
            obs (dict): Reference observation.
            bound (Bound): Bound of RatioEstimator.
            th (float): Threshold value, default -13
        """
        bounds = {}
        udim = bound.udim
        idx_rec = []

        samples = post.sample(N, obs)
        v = samples["v"]
        u = post.prior.u(v)

        weights = samples["weights"]

        for part in partition:
            if part not in weights:
                raise KeyError
            #            if len(part) == 1:
            #                idx_rec.append(part[0])
            #            else:
            w = weights[part]
            mask = w / w.max() > np.exp(th)
            points = u[mask][:, part]
            b = BallsBound(points)
            bounds[part] = b

        #        if len(idx_rec) > 0:
        #            res = np.zeros((len(idx_rec, 2))
        #            res[:, 1] = 1.0
        #            part = tuple(idx_rec)
        #            for i in part:
        #                w = weights[(i,)]
        #                mask = w-w.max() > th
        #                p = points[mask,i]
        #                [p.min(), p.max()]
        #            points = points[:,part]
        #            #bounds[part] = RectangleBound.from_RatioEstimator(
        #            #    rc, obs, bound, th=th, part=part
        # )

        return cls(bounds, udim)

    def state_dict(self):
        state_dict = dict(
            tag="CompositBound",
            udim=self._udim,
            bounds={k: v.state_dict() for k, v in self._bounds.items()},
        )
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict):
        bounds = {k: Bound.from_state_dict(v) for k, v in state_dict["bounds"].items()}
        udim = state_dict["udim"]
        return cls(bounds, udim)
