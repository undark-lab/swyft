from typing import Callable

import numpy as np
import torch

from swyft.bounds.bounds import Bound, UnitCubeBound
from swyft.types import PathType


class TruncatedPrior:
    """Truncated prior.

    Args:
        prior: Parameter prior
        bound: Bound object

    .. note::
        The truncated prior is defined through a swyft.Bound object, which
        sample from (subregions of) the hypercube, with swyft.Prior, which maps
        the samples onto parameters of interest.
    """

    def __init__(
        self,
        prior: "Prior",
        bound: Bound,
    ) -> None:
        """Instantiate truncated prior (combination of prior and bound).

        Args:
            prior: Prior object.
            bound: Bound on hypercube.  Set 'None' for untruncated priors.
        """
        self.prior = prior
        if bound is None:
            bound = UnitCubeBound(prior.zdim)
        self.bound = bound

    def sample(self, N: int) -> np.ndarray:
        """Sample from bounded prior.

        Args:
            N: Number of samples to return

        Returns:
            Samples: (N, zdim)
        """
        u = self.bound.sample(N)
        return self.prior.v(u)

    def log_prob(self, v: np.ndarray) -> np.ndarray:
        """Evaluate log probability of pdf.

        Args:
            v: (N, zdim) parameter points.

        Returns:
            log_prob: (N,)
        """
        u = self.prior.u(v)
        b = np.where(u.sum(axis=-1) == np.inf, 0.0, self.bound(u))
        log_prob = np.where(
            b == 0.0,
            -np.inf,
            self.prior.log_prob(v).sum(axis=-1) - np.log(self.bound.volume),
        )
        return log_prob

    def state_dict(self) -> dict:
        return dict(prior=self.prior.state_dict(), bound=self.bound.state_dict())

    @classmethod
    def from_state_dict(cls, state_dict: dict):
        prior = Prior.from_state_dict(state_dict["prior"])
        bound = Bound.from_state_dict(state_dict["bound"])
        return cls(prior, bound)

    @classmethod
    def load(cls, filename: PathType):
        sd = torch.load(filename)
        return cls.from_state_dict(sd)

    def save(self, filename: PathType) -> None:
        sd = self.state_dict()
        torch.save(sd, filename)


class Prior:
    def __init__(
        self,
        uv: Callable,
        zdim: int,
        n: int = 10000,
    ) -> None:
        r"""1-dim parameter prior.

        Args:
            uv: Function u->v
            zdim: Number of parameters
            n: Number of discretization points.

        .. note::
            The prior is defined through the mapping :math:`u\to v`, from the
            Uniform distribution, :math:`u\sim \text{Unif}(0, 1)` onto the
            parameters of interest, :math:`v`.  This mapping corresponds to the
            inverse cummulative distribution function, and is internally used
            to perform inverse transform sampling.  Sampling happens in the
            swyft.Bound object.

        .. warning::
            Internally the mapping u -> v is tabulated on a linear grid on the
            interval [0, 1], with `n` grid points. In extreme cases, this can
            lead to approximation errors that can be mitigated by increasing
            `n`.
        """
        self._zdim = zdim
        self._grid = np.linspace(0, 1.0, n)
        self._table = self._generate_table(uv, self._grid, zdim)

    @staticmethod
    def _generate_table(uv: Callable, grid: np.ndarray, zdim: int) -> np.ndarray:
        table = []
        for x in grid:
            table.append(uv(np.ones(zdim) * x))
        return np.array(table).T

    @property
    def zdim(self) -> int:
        """Number of parameters."""
        return self._zdim

    def u(self, v: np.ndarray) -> np.ndarray:
        """Map onto hypercube: v -> u

        Args:
            v: (N, zdim) physical parameter array

        Returns:
            u: (N, zdim) hypercube parameter array
        """
        u = np.empty_like(v)
        for i in range(self._zdim):
            u[:, i] = np.interp(
                v[:, i], self._table[i], self._grid, left=np.inf, right=np.inf
            )
        return u

    def v(self, u: np.ndarray) -> np.ndarray:
        """Map from hypercube: u -> v

        Args:
            u: (N, zdim) hypercube parameter array

        Returns:
            v: (N, zdim) physical parameter array
        """
        v = np.empty_like(u)
        for i in range(self._zdim):
            v[:, i] = np.interp(
                u[:, i], self._grid, self._table[i], left=np.inf, right=np.inf
            )
        return v

    def log_prob(self, v: np.ndarray, du: float = 1e-6) -> np.ndarray:
        """Log probability.

        Args:
            v: (N, zdim) physical parameter array
            du: Step-size of numerical derivatives

        Returns:
            log_prob: (N, zdim) factors of pdf
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

    def state_dict(self) -> dict:
        return dict(table=self._table, grid=self._grid, zdim=self._zdim)

    @classmethod
    def from_state_dict(cls, state_dict: dict):
        obj = cls.__new__(cls)
        obj._zdim = state_dict["zdim"]
        obj._grid = state_dict["grid"]
        obj._table = state_dict["table"]
        return obj

    @classmethod
    def load(cls, filename: PathType):
        sd = torch.load(filename)
        return cls.from_state_dict(sd)

    def save(self, filename: PathType) -> None:
        sd = self.state_dict()
        torch.save(sd, filename)
