from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch

from swyft.marginals.bounds import BallsBound, Bound, UnitCubeBound
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

class Prior(BoundedPrior):
    def __init__(self, ptrans):
        bound = UnitCubeBound(ptrans.ndim)
        super().__init__(ptrans, bound)

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

    @property
    def ndim(self):
        return self._ndim

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
