from typing import Callable

import numpy as np
import torch

from .bounds import Bound, UnitCubeBound


class Prior:
    """Prior with bounds."""

    def __init__(self, ptrans, bound=None):
        """Instantiate Prior.

        Args:
            ptrans (PriorTransform): Map from hypercube to physical parameters.
            bound (Bound): Bound on hypercube.
        """
        self.ptrans = ptrans
        if bound is None:
            bound = UnitCubeBound(ptrans.zdim)
        self.bound = bound

    def sample(self, N):
        """Sample from bounded prior.

        Args:
            N (int): Number of samples to return

        Returns:
            Samples (np.ndarray), (N, zdim)
        """
        u = self.bound.sample(N)
        return self.ptrans.v(u)

    def log_prob(self, v):
        """Evaluate log probability of pdf.

        Args:
            v (2-dim np.ndarray): (N, zdim) parameter points.

        Returns:
            log_prob (np.ndarray, (N,))
        """
        u = self.ptrans.u(v)
        b = np.where(u.sum(axis=-1) == np.inf, 0.0, self.bound(u))
        log_prob = np.where(
            b == 0.0,
            -np.inf,
            self.ptrans.log_prob(v).sum(axis=-1) - np.log(self.bound.volume),
        )
        return log_prob

    def state_dict(self):
        return dict(ptrans=self.ptrans.state_dict(), bound=self.bound.state_dict())

    def rebounded(self, bound):
        return Prior(self.ptrans, bound)

    @classmethod
    def from_uv(cls, uv, zdim, bound=None, n=10000):
        ptrans = PriorTransform(uv, zdim, n=n)
        return cls(ptrans, bound=bound)

    @classmethod
    def from_state_dict(cls, state_dict):
        if "grid" in state_dict["ptrans"]:
            ptrans = PriorTransform.from_state_dict(state_dict["ptrans"])
        else:
            ptrans = CustomTransform.from_state_dict(state_dict["ptrans"])
        bound = Bound.from_state_dict(state_dict["bound"])
        return Prior(ptrans, bound)

    @classmethod
    def load(cls, filename):
        sd = torch.load(filename)
        return cls.from_state_dict(sd)

    def save(self, filename):
        sd = self.state_dict()
        torch.save(sd, filename)


class CustomTransform:
    def __init__(self, u: Callable, v: Callable, log_prob: Callable, zdim: int) -> None:
        """Specify the forward transform (cdf, u), backward transform (ppf, c), and the logpdf(v)."""
        self._u = u
        self._v = v
        self._log_prob = log_prob
        self.zdim = zdim

    def u(self, v):
        """The cumulative distribution function."""
        return self._u(v)

    def v(self, u):
        """The percent point function, inverse cdf, or quantile function."""
        return self._v(u)

    def log_prob(self, v):
        """log pdf"""
        return self._log_prob(v)

    def state_dict(self):
        return dict(u=self._u, v=self._v, log_prob=self._log_prob, zdim=self.zdim)

    @classmethod
    def from_state_dict(cls, state_dict):
        obj = cls.__new__(cls)
        obj._u = state_dict["u"]
        obj._v = state_dict["v"]
        obj._log_prob = state_dict["log_prob"]
        obj.zdim = state_dict["zdim"]
        return obj


class PriorTransform:
    def __init__(self, uv, zdim, n=10000):
        """Prior transformation object.  Maps hypercube on physical parameters.

        Args:
            uv (callable): Function u->v
            zdim (int): Number of parameters
            n (int): Number of discretization points.
        """
        self._zdim = zdim
        self._grid = np.linspace(0, 1.0, n)
        self._table = self._generate_table(uv, self._grid, zdim, n)

    @staticmethod
    def _generate_table(uv, grid, zdim, n):
        table = []
        for x in grid:
            table.append(uv(np.ones(zdim) * x))
        return np.array(table).T

    @property
    def zdim(self):
        """Number of parameters."""
        return self._zdim

    def u(self, v):
        """Map onto hypercube: v -> u

        Args:
            v (np.array): (N, zdim) physical parameter array

        Returns:
            u (np.array): (N, zdim) hypercube parameter array
        """
        u = np.empty_like(v)
        for i in range(self._zdim):
            u[:, i] = np.interp(
                v[:, i], self._table[i], self._grid, left=np.inf, right=np.inf
            )
        return u

    def v(self, u):
        """Map from hypercube: u -> v

        Args:
            u (np.array): (N, zdim) hypercube parameter array

        Returns:
            v (np.array): (N, zdim) physical parameter array
        """
        v = np.empty_like(u)
        for i in range(self._zdim):
            v[:, i] = np.interp(
                u[:, i], self._grid, self._table[i], left=np.inf, right=np.inf
            )
        return v

    def log_prob(self, v, du=1e-6):
        """Log probability.

        Args:
            v (np.array): (N, zdim) physical parameter array
            du (float): Step-size of numerical derivatives

        Returns:
            log_prob (np.array): (N, zdim) factors of pdf
        """
        dv = np.empty_like(v)
        u = self.u(v)
        for i in range(self._zdim):
            dv[:, i] = np.interp(
                u[:, i] + (du / 2), self._grid, self._table[i], left=None, right=None
            )
            dv[:, i] -= np.interp(
                u[:, i] - (du / 2), self._grid, self._table[i], left=None, right=None
            )
        log_prob = np.where(u == np.inf, -np.inf, np.log(du) - np.log(dv + 1e-300))
        return log_prob

    def state_dict(self):
        return dict(table=self._table, grid=self._grid, zdim=self._zdim)

    @classmethod
    def from_state_dict(cls, state_dict):
        obj = cls.__new__(cls)
        obj._zdim = state_dict["zdim"]
        obj._grid = state_dict["grid"]
        obj._table = state_dict["table"]
        return obj

    @classmethod
    def load(cls, filename):
        sd = torch.load(filename)
        return cls.from_state_dict(sd)

    def save(self, filename):
        sd = self.state_dict()
        torch.save(sd, filename)
