from typing import (
    Callable,
    Dict,
    Hashable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from dataclasses import dataclass
import numpy as np
import torch
import swyft
import swyft.lightning.utils
import scipy.stats


@dataclass
class RectangleBounds:
    """Dataclass for storing rectangular bounds.

    Args:
        bounds: Bounds
        parnames: Parameter names
    """

    bounds: torch.Tensor
    parnames: np.array


def _rect_bounds_from_tensors(
    params: torch.Tensor, logratios: torch.Tensor, threshold=1e-6
):
    """Takes parameter array and logratios array and extracts rectangular bounds

    Args:
        params: (Nsamples, *Nparams, Ndim)
        logratios: (Nsamples, *Nparams)

    Returns:
        np.Tensor: (*Nparams, Ndim, 2)
    """
    # Generate mask (all points with maximum likelihood ratio above threshold are kept)
    mask = logratios - logratios.max(axis=0).values > np.log(threshold)
    par_max = params.max(dim=0).values
    par_min = params.min(dim=0).values
    constr_min = torch.where(mask.unsqueeze(-1), params, par_max).min(dim=0).values
    constr_max = torch.where(mask.unsqueeze(-1), params, par_min).max(dim=0).values
    return torch.stack([constr_min, constr_max], dim=-1)


def get_rect_bounds(logratios, threshold: float = 1e-6):
    """Extract rectangular bounds.

    Args:
        lrs_coll: Collection of LogRatioSample objects
        threshold: Threshold value for likelihood ratios.
    """
    logratios = swyft.lightning.utils._collection_mask(
        logratios, lambda x: isinstance(x, swyft.lightning.core.LogRatioSamples)
    )

    def map_fn(logratios):
        bounds = _rect_bounds_from_tensors(
            logratios.params, logratios.logratios, threshold=threshold
        )
        return RectangleBounds(bounds=bounds, parnames=logratios.parnames)

    return swyft.lightning.utils._collection_map(logratios, lambda x: map_fn(x))


class RectBoundSampler:
    """Sampler for rectangular bound regions.

    Args:
        distr: Description of probability distribution, or list of distributions
        bounds:
    """

    def __init__(self, distr, bounds=None):
        if not isinstance(distr, list):
            distr = [distr]
        self._distr = distr
        self._u = []
        i = 0
        for d in distr:
            if isinstance(d, scipy.stats._distn_infrastructure.rv_frozen):
                s = np.atleast_1d(d.rvs())
                j = len(s)
                u_low = s * 0.0 if bounds is None else d.cdf(bounds[i : i + j, ..., 0])
                u_high = (
                    s * 0.0 + 1.0
                    if bounds is None
                    else d.cdf(bounds[i : i + j, ..., 1])
                )
                self._u.append([u_low, u_high])
                i += j
            else:
                raise TypeError

    def __call__(self):
        result = []
        for i, d in enumerate(self._distr):
            if isinstance(d, scipy.stats._distn_infrastructure.rv_frozen):
                #                u = scipy.stats.uniform(
                #                    loc=self._u[i][0], scale=self._u[i][1] - self._u[i][0]
                #                ).rvs()
                u = np.random.rand(*self._u[i][0].shape)
                u *= self._u[i][1] - self._u[i][0]
                u += self._u[i][0]
                result.append(np.atleast_1d(d.ppf(u)))
            else:
                raise TypeError
        return np.hstack(result)


def collect_rect_bounds(
    lrs_coll, parname: str, parshape: tuple, threshold: float = 1e-6
):
    """Collect rectangular bounds for a parameter of interest.

    Args:
        lrs_coll: Collection of LogRatioSamples
        parname: Name of parameter vector/array of interest
        parshape: Shape of parameter vector/array
        threshold: Likelihood-ratio selection threshold
    """
    bounds = swyft.lightning.bounds.get_rect_bounds(lrs_coll, threshold=threshold)
    box = []
    for i in range(*parshape):
        try:
            i0, i1 = swyft.lightning.utils.param_select(
                bounds.parnames, [parname + "[%i]" % i]
            )
            b = bounds.bounds[i0][i1]
        except swyft.lightning.utils.SwyftParameterError:
            b = torch.tensor([[-np.inf, +np.inf]])
        box.append(b)
    return torch.cat(box)


# @dataclass
# class MeanStd:
#    """Store mean and standard deviation"""
#    mean: torch.Tensor
#    std: torch.Tensor
#
#    def from_samples(samples, weights = None):
#        """
#        Estimate mean and std deviation of samples by averaging over first dimension.
#        Supports weights>=0 with weights.shape = samples.shape
#        """
#        if weights is None:
#            weights = torch.ones_like(samples)
#        mean = (samples*weights).sum(axis=0)/weights.sum(axis=0)
#        res = samples - mean
#        var = (res*res*weights).sum(axis=0)/weights.sum(axis=0)
#        return MeanStd(mean = mean, std = var**0.5)

# def get_1d_rect_bounds(samples, th = 1e-6):
#    bounds = {}
#    r = samples.logratios
#    r = r - r.max(axis=0).values  # subtract peak
#    p = samples.params
#    all_max = p.max(dim=0).values
#    all_min = p.min(dim=0).values
#    constr_min = torch.where(r > np.log(th), p, all_max).min(dim=0).values
#    constr_max = torch.where(r > np.log(th), p, all_min).max(dim=0).values
#    #bound = torch.stack([constr_min, constr_max], dim = -1)
#    bound = RectangleBound(constr_min, constr_max)
#    return bound
