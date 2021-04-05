import numpy as np
from sklearn.neighbors import BallTree
from swyft.inference.ratioestimation import IsolatedRatio

# FIXME: Add docstring
class Bound:
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
        tag = state_dict['tag']
        if tag == 'UnitCubeBound':
            return UnitCubeBound.from_state_dict(state_dict)
        elif tag == 'RectangleBound':
            return RectangleBound.from_state_dict(state_dict)
        elif tag == 'BallsBound':
            return BallsBound.from_state_dict(state_dict)
        elif tag == 'CompositBound':
            return CompositBound.from_state_dict(state_dict)
        else:
            raise KeyError

    @classmethod
    def from_RatioCollection(cls, rc, obs, th, udim):
        return CompositBound.from_RatioCollection(rc, obs, th, udim)


class UnitCubeBound(Bound):
    """The unit hypercube bound."""
    def __init__(self, udim):
        """Initialize unit hypercube bound.

        Args:
            udim (int): Number of parameters.
        """
        self._udim = udim
        self._volume = 1.

    @property
    def volume(self):
        return self._volume

    @property
    def udim(self):
        return self._udim

    def sample(self, N):
        return np.random.rand(N, self.udim)

    def __call__(self, u):
        b = np.where(u <= 1., np.where(u >= 0., 1., 0.), 0.)
        return b.prod(axis=-1)

    def state_dict(self):
        return dict(tag='UnitCubeBound', udim=self.udim)

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(state_dict['udim'])


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
        V = 1.
        for i in range(self.udim):
            V *= self._rec_bounds[i,1] - self._rec_bounds[i,0]
        return V

    @property
    def udim(self):
        return len(self._rec_bounds)
            
    def sample(self, N):
        u = np.random.rand(N, self.udim)
        for i in range(self.udim):
            u[:,i] *= self._rec_bounds[i,1]-self._rec_bounds[i,0]
            u[:,i] += self._rec_bounds[i,0]
        return u

    def __call__(self, u):
        m = np.ones(len(u))
        for k, v in self._rec_bounds.items():
            m *= np.where(u[:,k] >= v[0], np.where(u[:,k] <= v[1], 1., 0.), 0.)
        return m > 0.5

    @classmethod
    def from_RatioCollection(cls, rc, obs, bound, th = -13, n = 10000):
        """Generate new RectangleBound object based on RatioCollection.

        Args:
            rc (RatioCollection): RatioCollection to evaluate.
            obs (dict): Reference observation.
            bound (Bound): Bound of RatioCollection.
            th (float): Threshold value, default -13
            n (int): Number of random samples from bound to determine parameter boundaries.

        Note: All components of the RatioCollection will be used.  Avoid overlapping ratios.
        """
        udim = bound.udim
        u = bound.sample(n)
        ratios = rc.ratios(obs, u)
        res = np.zeros((udim, 2))
        res[:,1] = 1.
        for k, v in ratios.items():
            for i in k:
                us = u[:,i][v-v.max() > th]
                res[i, 0] = us.min()
                res[i, 1] = us.max()
        
        return cls(res)

    def state_dict(self):
        return dict(tag='RectangleBound', rec_bounds=self._rec_bounds)

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(state_dict['rec_bounds'])


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

    def __call__(self, u):
        u = u.reshape(len(u), -1)
        dist, ind = self.bt.query(u, k=1)
        return (dist < self.epsilon)[:, 0]

    @classmethod
    def from_IsolatedRatio(cls, ratio, obs, bound, n=10000, th=-13):
        """Generate new BallsBound object based on IsolatedRatio.

        Args:
            ratio (IsolatedRatio): Single ratio.
            obs (dict): Reference observation.
            bound (Bound): Bound of RatioCollection.
            th (float): Threshold value, default -13
            n (int): Number of random samples from bound to determine parameter boundaries.

        Note: All components of the RatioCollection will be used.  Avoid overlapping ratios.
        """
        u = bound.sample(n)
        masklist = {}
        r = ratio(u)
        mask = r.max() - r < -th
        ind_points = u[mask]
        return cls(ind_points)

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

    def __call__(self, u):
        res = []
        for k, v in self._bounds.items():
            r = v(u[:, np.array(k)])
            res.append(r)
        return sum(res) == len(res)

    @classmethod
    def from_RatioCollection(cls, rc, obs, bound, th = -13.):
        """Generate new CompositBound object based on RatioCollection.

        Args:
            rc (RatioCollection): RatioCollection to evaluate.
            obs (dict): Reference observation.
            bound (Bound): Bound of RatioCollection.
            th (float): Threshold value, default -13
        """
        bounds = {}
        # FIXME: Do something more clever here
        udim = bound.udim
        bounds[tuple(range(udim))] = RectangleBound.from_RatioCollection(rc, obs, bound, th = th)
        return cls(bounds, udim)

#        #rb = RectangleBound.from_RatioCollection(rc, obs, th, udim)
#        for comb in rc.param_list:
#            if len(comb) == 1:  # 1-dim case
#                bound = UnitCubeBound(1)
#                ratio = IsolatedRatio(rc, obs, comb, udim)
#                b = BallsBound.from_IsolatedRatio(ratio, obs, bound)
#            elif len(comb) == 2:  # 2-dim case
#                bound = UnitCubeBound(2)  
#                ratio = IsolatedRatio(rc, obs, comb, udim)
#                b = BallsBound.from_IsolatedRatio(ratio, obs, bound)
#            else:
#                raise NotImplementedError
#            bounds[comb] = b
#
#        return cls(bounds, udim)

    def state_dict(self):
        state_dict = dict(
                tag="CompositBound",
                udim=self._udim,
                bounds = {k: v.state_dict() for k, v in self._bounds.items()}
                )
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict):
        bounds = {k: Bound.from_state_dict(v) for k, v in state_dict['bounds'].items()}
        udim = state_dict['udim']
        return cls(bounds, udim)
