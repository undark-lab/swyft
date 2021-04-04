import numpy as np
from sklearn.neighbors import BallTree

# FIXME: Rename mask.py to bounds.py

class CompositBound:
    """Composition bound object in u space."""
    def __init__(self, bounds):
        """
        Args:
            bounds (dict): dictionary with entries {index_list: Bound}
        """
        self._bounds = bounds
        self.volume = self._get_volume(bounds)

    def sample(self, N):
        for key, value in self._bounds.items():
            print(key)
        raise NotImplementedError

    def __call__(self, u):
        raise NotImplementedError

    @staticmethod
    def _get_volume(bounds):
        V = 1.
        for key, value in bounds.items():
            V *= value.volume
        return V


class UnitCubeBound:
    def __init__(self, ndim):
        self.ndim = ndim
        self.volume = 1.

    def sample(self, N):
        return np.random.rand(N, self.ndim)

    def __call__(self, u):
        b = np.where(u <= 1., np.where(u >= 0., 1., 0.), 0.)
        return b.prod(axis=-1)


class BallsBound:
    def __init__(self, points, scale=1.0):
        """Simple mask based on coverage balls around inducing points.

        Args:
            points (Array): shape (num_points, n_dim)
            scale (float): Scale ball size (default 1.0)
        """
        assert len(points.shape) == 2
        self.X = points
        self.dim = self.X.shape[-1]
        self.bt = BallTree(self.X, leaf_size=2)
        self.epsilon = self._set_epsilon(self.X, self.bt, scale)
        self.volume = self._get_volume(self.X, self.epsilon, self.bt)

    @classmethod
    def from_state_dict(cls, state_dict):
        obj = cls.__new__(cls)
        obj.X = state_dict["points"]
        assert len(obj.X.shape) == 2
        obj.dim = obj.X.shape[-1]
        obj.epsilon = state_dict["epsilon"]
        obj.volume = state_dict["volume"]
        obj.bt = BallTree(obj.X, leaf_size=2)
        return obj

    def state_dict(self):
        return dict(
            masktype="BallsBound", points=self.X, epsilon=self.epsilon, volume=self.volume
        )

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
            print("WARNING: Rel volume uncertainty is:", rel)
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

    def __call__(self, points):
        """Evaluate points in mask."""
        points = points.reshape(len(points), -1)
        dist, ind = self.bt.query(points, k=1)
        return (dist < self.epsilon)[:, 0]

    def _initialize(self, obs, bound, ratio, N=10000, th=-7):
        raise NotImplementedError
        u = bound.sample(N)
        masklist = {}
        lnL = re.lnL(obs, pars)
        for k, v in self.to_cube(pars).items():
            mask = lnL[(k,)].max() - lnL[(k,)] < -th
            ind_points = v[mask].reshape(-1, 1)
            masklist[k] = BallsBound(ind_points)

    @classmethod
    def from_ratio(cls, bound, obs, ratio, N=10000, th=-7):
        u = bound.sample(N)
        masklist = {}
        r = ratio(u)
        mask = r.max() - r < -th
        print(len(mask), sum(mask))
        ind_points = u[mask]
        print(ind_points.shape)
        return cls(ind_points)


class ComboMask:
    def __init__(self, samplers):
        """Combination of lower dimensional masks.

        Args:
            samplers (dict): Dictionary of masks, mapping parameter names
            (strings or tuple of strings) to masks.

        Example:
            samplers = {"x": mask_1d, ("y", "z"): mask_2d}
        """
        self.samplers = samplers
        self.masks = None  # TODO: Implement additional masks

    def sample(self, N):
        result = {}
        for key in self.samplers.keys():
            if isinstance(key, str):
                result[key] = self.samplers[key].sample(N)[:, 0]
            elif isinstance(key, tuple):
                for i, k in enumerate(key):
                    result[k] = self.samplers[key].sample(N)[:, i]
            else:
                raise KeyError("ComboMask keys must be strings or string tuples.")
        return result

    @property
    def volume(self):
        volume = 1.0
        for key in self.samplers:
            volume *= self.samplers[key].volume
        return volume

    def __call__(self, X):
        res = []
        for k in self.samplers:
            r = self.samplers[k](X[k])
            res.append(r)
        return sum(res) == len(res)

    @classmethod
    def from_state_dict(cls, state_dict):
        samplers = {}
        for key, value in state_dict["samplers"].items():
            samplers[key] = BallsBound.from_state_dict(value)
        return cls(samplers)

    def state_dict(self):
        samplers = {}
        for key, value in self.samplers.items():
            samplers[key] = value.state_dict()
        return dict(samplers=samplers)

class CompositBound:
    def __init__(self, bounds, zdim):
        """
        Args:
            bounds (dict): Dictionary mapping indices like (0, 3) etc --> bounds
            zdim (int): Length of parameter vector.
        """
        self._bounds = bounds
        self._zdim = zdim
        self.mask = self._gen_mask()

    def sample(self, N):
        """Sample from composit bounds object

        Returns:
            Returns samples. Returns -1 where no samples where drawn.
        """
        results = -np.ones(N, self._zdim)
        for k, v in self._bounds.items():
            results[:, np.array(k)] = v.sample(N)
        return results

    @property
    def volume(self):
        volume = 1.0
        for k, v in self._bounds.items():
            volume *= v.volume
        return volume

    def __call__(self, z):
        res = []
        for k, v in self._bounds.items():
            r = v(z[:, np.array(k)])
            res.append(r)
        return sum(res) == len(res)

#    @classmethod
#    def from_state_dict(cls, state_dict):
#        samplers = {}
#        for key, value in state_dict["samplers"].items():
#            samplers[key] = BallsBound.from_state_dict(value)
#        return cls(samplers)
#
#    def state_dict(self):
#        samplers = {}
#        for key, value in self.samplers.items():
#            samplers[key] = value.state_dict()
#        return dict(samplers=samplers)
