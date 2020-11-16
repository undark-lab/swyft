# pylint: disable=no-member, not-callable, undefined-variable
from functools import cached_property
import re

import numpy as np

from .types import Shape, Union, Sequence, Tensor
from .eval import eval_net
from .utils import array_to_tensor


class Intensity:
    """Intensity function constructed from a d-dim FactorMask."""

    def __init__(self, expected_n: int, factor_mask):
        """Creates a expected_n intensity function over the FactorMask.

        Args:
            expected_n (int): expected target number of samples
            factor_mask (FactorMask): d-dim FactorMask
        """
        self.expected_n = expected_n
        self.factor_mask = factor_mask
        self._area = None

    def __call__(self, z):
        return self.factor_mask(z) / self.area * self.expected_n

    def __repr__(self):
        return f"{self.__class__.__name__}({self.expected_n}, {repr(self.factor_mask)})"

    @cached_property
    def area(self):
        if self._area is None:
            return self.factor_mask.area()
        else:
            return self._area

    def sample(self, n=None):
        if n is None:
            n = np.random.poisson(self.expected_n)
        return self.factor_mask.sample(n)

    @property
    def intervals(self):
        return self.factor_mask.intervals

    @property
    def zdim(self):
        return self.factor_mask.dim


def get_unit_intensity(expected_n: int, dim: int):
    """Creates a expected_n intensity function over the unit dim hypercube.

    Args:
        expected_n (int): expected target number of samples
        dim (int): dim of hypercube
    """
    factor_mask = get_unit_factor_mask(dim)
    return Intensity(expected_n, factor_mask)


def get_constrained_intensity(
    expected_n: int,
    ratio_estimator,
    x0: Tensor,
    threshold: float,
    factor_mask=None,
    samples: int = 10000,
):
    """Creates a expected_n intensity function constrained to ratio estimates over the treshold.
    The constraints are generated by sampling within a region. It is possible to restrict that search region using factor_mask.

    Args:
        expected_n (int): expected target number of samples
        ratio_estimator (RatioEstimator): takes signature ratio_estimator(x, z) and returns a likelihood ratio
        x0 (tensor): true observation
        threshold (float): threshold for constraint
        factor_mask (Intensity): (Optional) factorized region, in which to search for constraints
        samples (int): number of samples to produce inorder to find the constrained intervals
    """
    device = ratio_estimator.device
    zdim = ratio_estimator.zdim
    net = ratio_estimator.net
    if factor_mask is None:
        factor_mask = get_unit_factor_mask(zdim)

    # TODO make this into something which does not depend on ratio estimator

    z = array_to_tensor(factor_mask.sample(samples), device=device).unsqueeze(-1)
    x0 = array_to_tensor(x0, device=device)
    ratios = eval_net(net, x0, z)
    z = z.cpu().numpy()[:, :, 0]
    ratios = ratios.cpu().numpy()

    intervals_list = []
    for i in range(zdim):
        ratios_max = ratios[:, i].max()
        intervals = construct_intervals(
            z[:, i], ratios[:, i] - ratios_max - np.log(threshold)
        )
        intervals_list.append(intervals)

    factor_mask = get_factor_mask_from_intervals(intervals_list)
    print("Constrained posterior area:", factor_mask.area())
    return Intensity(expected_n, factor_mask)


def construct_intervals(x, y):
    r"""Let x be the base space and let y be a scalar field over x. Get intervals \in x where y \geq 0.

    Returns a list of lists organized by:
    [[upcrossing, downcrossing], [upcrossing, downcrossing], ...]
    where upcrossing means y crosses from below zero to above zero in that region,
    and downcrossing means y crosses from above zero to below zero in that region.
    """
    assert x.shape == y.shape
    max_index = len(x) - 1

    indices = np.argsort(x)
    x = x[indices]
    y = y[indices]
    m = np.where(y >= 0.0, 1.0, 0.0)
    m = m[1:] - m[:-1]
    upcrossings = np.argwhere(m == 1.0)[:, 0]
    downcrossings = np.argwhere(m == -1.0)[:, 0]

    # TODO a y which is entirely above zero will return the whole interval. This is bad.
    # No crossings --> return entire interval
    if len(upcrossings) == 0 and len(downcrossings) == 0:
        return [[x[0], x[-1]]]

    if len(upcrossings) - len(downcrossings) == 1:
        # One more upcrossing than downcrossing
        # --> Treat right end as downcrossing
        downcrossings = np.append(downcrossings, max_index)
    elif len(upcrossings) - len(downcrossings) == -1:
        # One more downcrossing than upcrossing
        # --> Treat left end as upcrossing
        upcrossings = np.append(0, upcrossings)
    elif len(upcrossings) == len(downcrossings):
        pass
    else:
        raise ValueError(
            "It should be impossible to have two more crossings of one type, than the other."
        )

    intervals = []
    for down, up in zip(downcrossings, upcrossings):
        if up > down:
            intervals.append([x[down], x[up]])
        elif up < down:
            intervals.append([x[up], x[down]])
        elif up < 0 or down < 0:
            raise ValueError(
                "Constructing intervals with negative indexes is not allowed."
            )
        else:
            raise ValueError("Cannot have an up and down crossing at the same index.")
    return intervals


class Mask1d:
    """A 1-dim multi-interval based mask class."""

    def __init__(self, intervals: Sequence[Sequence[float]]):
        """One dimensional mask which indicates whether or not points lie within the region.

        Args:
            intervals (list of sorted tuples of floats): (lower, upper) bounds on regions which yield one upon evaluation. intervals with very small length are ignored.
        """
        increasing = lambda x, y: x < y
        zero_measure = lambda x, y: np.isclose(x, y, rtol=1e-8, atol=1e-10)
        measurable_intervals = [[x, y] for x, y in intervals if not zero_measure(x, y)]
        for x, y in measurable_intervals:
            assert increasing(
                x, y
            ), "One of the intervals was decreasing. They need to be (lower bound, upper bound)."
        self.intervals = np.asarray(measurable_intervals)  # n x 2 matrix

    def __call__(self, z):
        """Returns 1. if inside interval, otherwise 0."""
        m = np.zeros_like(z)
        for z0, z1 in self.intervals:
            zs_lt_z1 = z <= z1
            zs_gt_z0 = z >= z0
            ones_at_lt_z1 = np.where(zs_lt_z1, 1.0, 0.0)
            m += np.where(zs_gt_z0, ones_at_lt_z1, 0.0)
        assert not any(m > 1.0), "Overlapping intervals."
        return m

    def __repr__(self):
        return f"{self.__class__.__name__}({self.intervals.tolist()})"

    def area(self):
        """Combined length of all intervals (AKAK 1-dim area)."""
        return (self.intervals[:, 1] - self.intervals[:, 0]).sum()

    def sample(self, n):
        p = self.intervals[:, 1] - self.intervals[:, 0]
        p /= p.sum()
        i = np.random.choice(len(p), size=n, replace=True, p=p)
        w = np.random.rand(n)
        z = self.intervals[i, 0] + w * (self.intervals[i, 1] - self.intervals[i, 0])
        return z


class FactorMask:
    """A d-dim factorized mask."""

    def __init__(self, masks: Sequence):
        self.masks = masks
        self.dim = len(masks)

    def __call__(self, z):
        m = [self.masks[i](z[:, i]) for i in range(self.dim)]
        m = np.array(m).prod(axis=0)
        return m

    def __repr__(self):
        masks_as_string = ", ".join([repr(mask) for mask in self.masks])
        return f"{self.__class__.__name__}([{masks_as_string}])"

    def area(self):
        m = [self.masks[i].area() for i in range(self.dim)]
        return np.array(m).prod()

    def sample(self, n):
        z = np.empty((n, self.dim))
        for i in range(self.dim):
            z[:, i] = self.masks[i].sample(n)
        return z

    @property
    def intervals(self):
        return [mask.intervals for mask in self.masks]


def get_unit_factor_mask(dim):
    """Returns FactorMask on the dim unit hypercube."""
    return FactorMask([Mask1d([[0.0, 1.0]]) for _ in range(dim)])


def get_factor_mask_from_intervals(intervals):
    """Returns FactorMask from intervals. Intervals must have ndim == 2 or 3."""
    intervals = np.asarray(intervals)
    assert intervals.ndim in [2, 3]
    if intervals.ndim == 2:
        return FactorMask([Mask1d(intervals)])
    elif intervals.ndim == 3:
        return FactorMask([Mask1d(set_of_intervals) for set_of_intervals in intervals])
    else:
        raise ValueError(
            "Intervals must be a list of intervals or a list of a list of intervals."
        )