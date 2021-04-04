from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch

from swyft.marginals.bounds import BallsBound, Bound
from swyft.types import Array, PriorConfig
from swyft.utils import array_to_tensor, depth, tensor_to_array


class BoundedPrior:
    def __init__(self, ptrans, bound):
        self.ptrans = ptrans
        self.bound = bound

    def sample(self, N): 
        u = self.bound.sample(N)
        return self.ptrans.v(u)

    def log_prob(self, v): 
        u = self.ptrans.u(v)
        b = np.where(u.sum(axis=-1) == np.inf, 0., self.bound(u))
        log_prob = np.where(b == 0., -np.inf, self.ptrans.log_prob(v).sum(axis=-1) - np.log(self.bound.volume))
        return log_prob

    def state_dict(self):
        return dict(ptrans=self.ptrans.state_dict(), bound=self.bound.state_dict())

    @classmethod
    def from_state_dict(cls, state_dict):
        ptrans = PriorTransform.from_state_dict(state_dict['ptrans'])
        bound = Bound.from_state_dict(state_dict['bound'])
        return cls(ptrans, bound)


class PriorTransform:
    def __init__(self, ptrans, ndim, n_steps= 10000):
        """Prior transformation object.  Maps hypercube on physical parameters.

        Args
        """
        self._ndim = ndim
        self._grid = np.linspace(0, 1., n_steps)
        self._table = self._generate_table(ptrans, self._grid, ndim, n_steps)

    @staticmethod
    def _generate_table(ptrans, grid, ndim, n_steps):
        table = []
        for x in grid:
            table.append(ptrans(np.ones(ndim)*x))
        return np.array(table).T

    def u(self, v):
        """CDF, mapping v->u"""
        u = np.empty_like(v)
        for i in range(self._ndim):
            u[:,i] = np.interp(v[:,i], self._table[i], self._grid, left = np.inf, right = np.inf)
        return u

    def v(self, u):
        """Inverse CDF, mapping u->v"""
        v = np.empty_like(u)
        for i in range(self._ndim):
            v[:,i] = np.interp(u[:,i], self._grid, self._table[i], left = np.inf, right = np.inf)
        return v

    def log_prob(self, v, du = 1e-6):
        """log prior(v)"""
        dv = np.empty_like(v)
        u = self.u(v)
        for i in range(self._ndim):
            dv[:,i] = np.interp(u[:,i] +(du/2), self._grid, self._table[i], left = None, right = None)
            dv[:,i] -= np.interp(u[:,i]-(du/2), self._grid, self._table[i], left = None, right = None)
        log_prob = np.where(u == np.inf, -np.inf, np.log(du) - np.log(dv+1e-300))
        return log_prob

    def state_dict(self):
        return dict(table=self._table, grid=self._grid, ndim=self._ndim)

    @classmethod
    def from_state_dict(cls, state_dict):
        obj = cls.__new__(cls)
        obj._ndim = state_dict['ndim']
        obj._grid = state_dict['grid']
        obj._table = state_dict['table']
        return obj


#class Prior1d:
#    def __init__(self, tag, *args):
#        self.tag = tag
#        self.args = args
#        if tag == "normal":
#            loc, scale = args[0], args[1]
#            self.prior = torch.distributions.Normal(loc, scale)
#        elif tag == "uniform":
#            x0, x1 = args[0], args[1]
#            self.prior = torch.distributions.Uniform(x0, x1)
#        elif tag == "lognormal":
#            loc, scale = args[0], args[1]
#            self.prior = torch.distributions.LogNormal(loc, scale)
#        else:
#            raise KeyError("Tag unknown")
#
#    def sample(self, N):
#        return tensor_to_array(self.prior.sample((N,)), np.float64)
#
#    def log_prob(self, value):
#        return self.prior.log_prob(value).numpy()
#
#    def to_cube(self, value):
#        return self.prior.cdf(torch.tensor(value)).numpy()
#
#    def from_cube(self, value):
#        return self.prior.icdf(torch.tensor(value)).numpy()
#
#    def state_dict(self):
#        return dict(tag=self.tag, args=self.args)
#
#    @classmethod
#    def from_state_dict(cls, state_dict):
#        return cls(state_dict["tag"], *state_dict["args"])


class Prior:
    """Accomodates the completely factorized prior, log_prob, sampling, and 'volume' calculations."""

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

    # FIXME: Mask / constraint should be an argument
    def sample(self, N):
        if self.mask is None:
            return self._sample_from_priors(N)
        else:
            samples = self.mask.sample(N)
            return self.from_cube(samples)

    def _sample_from_priors(self, N):
        result = {}
        for key, value in self.priors.items():
            result[key] = tensor_to_array(value.sample(N))
        return result

    # FIXME: Do we still need the volume?
    def volume(self):
        if self.mask is None:
            return 1.0
        else:
            return self.mask.volume

    # FIXME: Mask / constraint should be an argument
    # FIXME: Do we still need masked log_prob?
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

    # FIXME: Do we still need this?
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

    # FIXME: We do not save the mask, just unmasked priors.
    def state_dict(self):
        mask_dict = None if self.mask is None else self.mask.state_dict()
        return dict(prior_config=self.prior_config, mask=mask_dict)

    # FIXME: We do not restore the mask, just unmasked priors.
    @classmethod
    def from_state_dict(cls, state_dict):
        mask = (
            None
            if state_dict["mask"] is None
            else ComboMask.from_state_dict(state_dict["mask"])
        )
        return cls(state_dict["prior_config"], mask=mask)

    # FIXME: This needs to be extended to high-dim posteriors
    # Right now the logic is :
    # - masks don't have a NN inside, and can be stored easily
    # - masks can be sampled from easily
    # - masks are defined through those samples
    # Changes required:
    # - Constrained priors cannot be saved anymore, only priors can
    # - We can sample from a constrained prior by providing a network
    def get_masked(self, obs, re, N=10000, th=-7):
        if re is None:
            return self
        pars = self.sample(N)
        masklist = {}
        lnL = re.lnL(obs, pars)
        for k, v in self.to_cube(pars).items():
            mask = lnL[(k,)].max() - lnL[(k,)] < -th
            ind_points = v[mask].reshape(-1, 1)
            masklist[k] = BallsBound(ind_points)
        mask = ComboMask(masklist)
        return Prior(self.prior_config, mask=mask)
