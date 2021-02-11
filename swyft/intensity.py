# pylint: disable=no-member, not-callable, undefined-variable
import numpy as np
import torch
from sklearn.neighbors import BallTree

from .types import (
    Array,
    Device,
    Dict,
    Optional,
    PathType,
    PriorConfig,
    Sequence,
    Shape,
    Tuple,
    Union,
)
from .utils import verbosity


class BallMask:
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
            masktype="BallMask", points=self.X, epsilon=self.epsilon, volume=self.volume
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
            samplers[key] = BallMask.from_state_dict(value)
        return cls(samplers)

    def state_dict(self):
        samplers = {}
        for key, value in self.samplers.items():
            samplers[key] = value.state_dict()
        return dict(samplers=samplers)


class Prior1d:
    def __init__(self, tag, *args):
        self.tag = tag
        self.args = args
        if tag == "normal":
            loc, scale = args[0], args[1]
            self.prior = torch.distributions.Normal(loc, scale)
        elif tag == "uniform":
            x0, x1 = args[0], args[1]
            self.prior = torch.distributions.Uniform(x0, x1)
        elif tag == "lognormal":
            loc, scale = args[0], args[1]
            self.prior = torch.distributions.LogNormal(loc, scale)
        else:
            raise KeyError("Tag unknown")

    def sample(self, N):
        return self.prior.sample((N,)).type(torch.float64).numpy()

    def log_prob(self, value):
        return self.prior.log_prob(value).numpy()

    def to_cube(self, value):
        return self.prior.cdf(torch.tensor(value)).numpy()

    def from_cube(self, value):
        return self.prior.icdf(torch.tensor(value)).numpy()

    def state_dict(self):
        return dict(tag=self.tag, args=self.args)

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(state_dict["tag"], *state_dict["args"])


class Prior:
    def __init__(self, prior_config: PriorConfig, mask=None):
        self.prior_config = prior_config
        self.mask = mask

        self._setup_priors()

    def params(self):
        return self.prior_config.keys()

    def _setup_priors(self):
        result = {}
        for key, value in self.prior_config.items():
            result[key] = Prior1d(value[0], *value[1:])
        self.priors = result

    def sample(self, N):
        if self.mask is None:
            return self._sample_from_priors(N)
        else:
            samples = self.mask.sample(N)
            return self.from_cube(samples)

    def _sample_from_priors(self, N):
        result = {}
        for key, value in self.priors.items():
            result[key] = np.array(value.sample(N))
        return result

    def volume(self):
        if self.mask is None:
            return 1.0
        else:
            return self.mask.volume

    def log_prob(self, values, unmasked=False):
        log_prob_unmasked = {}
        for key, value in self.priors.items():
            x = torch.tensor(values[key])
            log_prob_unmasked[key] = value.log_prob(x)
        log_prob_unmasked_sum = sum(log_prob_unmasked.values())

        if self.mask is not None:
            cube_values = self.to_cube(values)
            m = self.mask(cube_values)
            log_prob_sum = np.where(
                m, log_prob_unmasked_sum - np.log(self.mask.volume), -np.inf
            )
        else:
            log_prob_sum = log_prob_unmasked_sum

        if unmasked:
            return log_prob_unmasked
        else:
            return log_prob_sum

    def factorized_log_prob(
        self,
        values: Dict[str, Array],
        targets: Union[str, Sequence[str], Sequence[Tuple[str]]],
        unmasked: bool = False,
    ):
        if depth(targets) == 0:
            targets = [(targets,)]
        elif depth(targets) == 1:
            targets = [tuple(targets)]

        log_prob_unmasked = {}
        for target in targets:
            relevant_log_probs = {key: self.priors[key].log_prob for key in target}
            relevant_params = {key: array_to_tensor(values[key]) for key in target}
            log_prob_unmasked[target] = sum(
                relevant_log_probs[key](relevant_params[key]) for key in target
            )

        if not unmasked and self.mask is not None:
            cube_values = self.to_cube(values)
            m = self.mask(cube_values)
            log_prob = {
                target: np.where(m, logp - np.log(self.mask.volume), -np.inf)
                for target, logp in log_prob_unmasked.items()
            }
        else:
            log_prob = log_prob_unmasked

        return log_prob

    def to_cube(self, X):
        out = {}
        for k, v in self.priors.items():
            out[k] = v.to_cube(X[k])
        return out

    def from_cube(self, values):
        result = {}
        for key, value in values.items():
            result[key] = np.array(self.priors[key].from_cube(value))
        return result

    def state_dict(self):
        mask_dict = None if self.mask is None else self.mask.state_dict()
        return dict(prior_config=self.prior_config, mask=mask_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        mask = (
            None
            if state_dict["mask"] is None
            else ComboMask.from_state_dict(state_dict["mask"])
        )
        return cls(state_dict["prior_config"], mask=mask)

    def get_masked(self, obs, re, N=10000, th=-7):
        if re is None:
            return self
        pars = self.sample(N)
        masklist = {}
        lnL = re.lnL(obs, pars)
        for k, v in self.to_cube(pars).items():
            mask = lnL[(k,)].max() - lnL[(k,)] < -th
            ind_points = v[mask].reshape(-1, 1)
            masklist[k] = BallMask(ind_points)
        mask = ComboMask(masklist)
        return Prior(self.prior_config, mask=mask)


class Intensity:
    def __init__(self, prior, mu):
        self.prior = prior
        self.mu = mu

    def sample(self):
        N = np.random.poisson(self.mu)
        return self.prior.sample(N)

    def __call__(self, values):
        return np.exp(self.prior.log_prob(values)) * self.mu

    @classmethod
    def from_state_dict(cls, state_dict):
        prior = Prior.from_state_dict(state_dict["prior"])
        mu = state_dict["mu"]
        return Intensity(prior, mu)

    def state_dict(self):
        prior = self.prior.state_dict()
        return dict(prior=prior, mu=self.mu)
