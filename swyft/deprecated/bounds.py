from __future__ import annotations

import logging
from typing import Callable, Optional, TypeVar

import numpy as np
from sklearn.neighbors import BallTree

from swyft.saveable import StateDictSaveable
from swyft.types import ObsType
from swyft.weightedmarginals import WeightedMarginalSamples

log = logging.getLogger(__name__)


BoundType = TypeVar("BoundType", bound="Bound")


class Bound(StateDictSaveable):
    """A bound region on the hypercube.

    .. note::
        The Bound object provides methods to sample from subregions of the
        hypercube, to evaluate the volume of the constrained region, and to
        evaluate the bound.
    """

    def __init__(self) -> None:
        pass

    @property
    def volume(self) -> float:
        """Volume of the bound region."""
        raise NotImplementedError

    @property
    def n_parameters(self) -> int:
        """Number of dimensions."""
        raise NotImplementedError

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample.

        Args:
            n_samples: Numbe of samples.

        Returns:
            s (n_samples x n_parameters)
        """
        raise NotImplementedError

    def __call__(self, u: np.ndarray):
        """Check whether parameters are within bounds.

        Args:
            u (n_samples x n_parameters): Parameters on hypercube
        """
        raise NotImplementedError

    # TODO can we do away with this thing? I think yes.
    @classmethod
    def from_state_dict(cls, state_dict) -> BoundType:
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

    @staticmethod
    def from_marginal_posterior(
        n_samples: int,
        observation: ObsType,
        marginal_posterior: "swyft.inference.marginalposterior.MarginalPosterior",
        threshold: float = -13.0,
        batch_size: Optional[int] = None,
    ) -> BoundType:
        """see CompositBound.from_marginal_posterior"""
        return CompositBound.from_marginal_posterior(
            n_samples,
            observation,
            marginal_posterior,
            threshold,
            batch_size,
        )


class UnitCubeBound(Bound, StateDictSaveable):
    """The unit hypercube bound."""

    def __init__(self, n_parameters):
        """Initialize unit hypercube bound.

        Args:
            n_parameters (int): Number of parameters.
        """
        self._n_parameters = n_parameters
        self._volume = 1.0

    @property
    def volume(self):
        """The volume of the constrained region."""
        return self._volume

    @property
    def n_parameters(self) -> int:
        return self._n_parameters

    def sample(self, n_samples):
        """Generate samples from the bound region.

        Args:
            n_samples (int): Number of samples
        """
        return np.random.rand(n_samples, self.n_parameters)

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
        return dict(tag="UnitCubeBound", n_parameters=self.n_parameters)

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(state_dict["n_parameters"])


class RectangleBound(Bound, StateDictSaveable):
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
        for i in range(self.n_parameters):
            V *= self._rec_bounds[i, 1] - self._rec_bounds[i, 0]
        return V

    @property
    def n_parameters(self):
        return len(self._rec_bounds)

    def sample(self, n_samples):
        u = np.random.rand(n_samples, self.n_parameters)
        for i in range(self.n_parameters):
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


class BallsBound(Bound, StateDictSaveable):
    def __init__(self, points, scale=1.0):
        """Simple mask based on coverage balls around inducing points.

        Args:
            points (Array): shape (num_points, n_dim)
            scale (float): Scale ball size (default 1.0)
        """
        assert len(points.shape) == 2
        self.X = points
        self._n_parameters = self.X.shape[-1]
        self.bt = BallTree(self.X, leaf_size=2)
        self.epsilon = self._set_epsilon(self.X, self.bt, scale)
        self._volume = self._get_volume(self.X, self.epsilon, self.bt)

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def n_parameters(self) -> int:
        return self._n_parameters

    @staticmethod
    def _set_epsilon(X, bt, scale):
        dims = X.shape[-1]
        k = [4, 5, 6]
        dist, ind = bt.query(X, k=k[dims - 1])  # 4th NN
        epsilon = np.median(dist[:, -1]) * scale * 1.5
        return epsilon

    @staticmethod
    def _get_volume(X, epsilon, bt):
        n_samples = 100
        vol_est = []
        d = X.shape[-1]
        area = {1: 2 * epsilon, 2: np.pi * epsilon**2}[d]
        for i in range(n_samples):
            n = np.random.randn(*X.shape)
            norm = (n**2).sum(axis=1) ** 0.5
            n = n / norm.reshape(-1, 1)
            r = np.random.rand(len(X)) ** (1 / d) * epsilon
            Y = X + n * r.reshape(-1, 1)
            in_bounds = ((Y >= 0.0) & (Y <= 1.0)).prod(axis=1, dtype="bool")
            Y = Y[in_bounds]
            counts = bt.query_radius(Y, epsilon, count_only=True)
            vol_est.append(area * sum(1.0 / counts))
        vol_est = np.array(vol_est)
        out, err = vol_est.mean(), vol_est.std() / np.sqrt(n_samples)
        rel = err / out
        if rel > 0.01:
            log.debug("WARNING: Rel volume uncertainty is %.4g" % rel)
        return out

    def sample(self, n_samples):
        counter = 0
        samples = []
        d = self.X.shape[-1]
        while counter < n_samples:
            n = np.random.randn(*self.X.shape)
            norm = (n**2).sum(axis=1) ** 0.5
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
        ind = np.random.choice(range(len(samples)), size=n_samples, replace=False)
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
        obj._n_parameters = obj.X.shape[-1]
        obj.epsilon = state_dict["epsilon"]
        obj._volume = state_dict["volume"]
        obj.bt = BallTree(obj.X, leaf_size=2)
        return obj


class CompositBound(Bound, StateDictSaveable):
    """Composit bound object. Product of multiple bounds."""

    def __init__(self, bounds_map, n_parameters):
        """
        Args:
            bounds_map (dict): Dictionary mapping indices like (0, 3) etc --> bounds
            n_parameters (int): Length of parameter vector.
        """
        self._bounds = bounds_map
        self._n_parameters = n_parameters

    def sample(self, n_samples):
        results = -np.ones((n_samples, self.n_parameters))
        for k, v in self._bounds.items():
            results[:, np.array(k)] = v.sample(n_samples)
        return results

    @property
    def volume(self) -> float:
        volume = 1.0
        for k, v in self._bounds.items():
            volume *= v.volume
        return volume

    @property
    def n_parameters(self) -> int:
        return self._n_parameters

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
    def from_weighted_samples(
        cls,
        weighted_samples: WeightedMarginalSamples,
        cdf: Callable,
        n_parameters: int,
        threshold: float,
    ) -> BoundType:
        """create a new bound object from weighted samples and the cdf

        Args:
            weighted_samples: log weighted samples
            cdf: transforms from v to u
            n_parameters: number of total parameters
            threshold: above which log weight do we bound? -13 is standard.

        Returns:
            a bound object based on the above
        """
        bounds = {}
        u = cdf(weighted_samples.v)
        for marginal_index in weighted_samples.marginal_indices:
            logw, _ = weighted_samples.get_logweight_marginal(marginal_index)
            mask = logw - logw.max() > threshold
            u_above_threshold = u[mask][:, marginal_index]
            bounds[marginal_index] = BallsBound(u_above_threshold)
        return cls(bounds, n_parameters)

    @classmethod
    def from_marginal_posterior(
        cls,
        n_samples: int,
        observation: ObsType,
        marginal_posterior: "swyft.inference.marginalposterior.MarginalPosterior",
        threshold: float,
        batch_size: Optional[int] = None,
    ) -> BoundType:
        """create a new bound object from a marginal posterior by sampling to estimate the log_prob contours

        Args:
            n_samples: number of samples to estimate with
            observation: single observation to define the bounds
            marginal_posterior: marginal posterior object
            threshold: above which log weight do we bound? -13 is standard
            batch_size: when evaluating the log_prob, what batch size to use

        Returns:
            a bound object based on the above
        """
        weighted_samples = marginal_posterior.weighted_sample(
            n_samples, observation, batch_size
        )
        return cls.from_weighted_samples(
            weighted_samples=weighted_samples,
            cdf=marginal_posterior.prior.cdf,
            n_parameters=marginal_posterior.prior.n_parameters,
            threshold=threshold,
        )

    def state_dict(self):
        state_dict = dict(
            tag="CompositBound",
            n_parameters=self.n_parameters,
            bounds={k: v.state_dict() for k, v in self._bounds.items()},
        )
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict):
        bounds = {k: Bound.from_state_dict(v) for k, v in state_dict["bounds"].items()}
        n_parameters = state_dict["n_parameters"]
        return cls(bounds, n_parameters)
