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


@dataclass
class MeanStd:
    """Store mean and standard deviation"""
    mean: torch.Tensor
    std: torch.Tensor

    def from_samples(samples, weights = None):
        """
        Estimate mean and std deviation of samples by averaging over first dimension.
        Supports weights>=0 with weights.shape = samples.shape
        """
        if weights is None:
            weights = torch.ones_like(samples)
        mean = (samples*weights).sum(axis=0)/weights.sum(axis=0)
        res = samples - mean
        var = (res*res*weights).sum(axis=0)/weights.sum(axis=0)
        return MeanStd(mean = mean, std = var**0.5)


@dataclass
class RectangleBound:
    low: torch.Tensor
    high: torch.Tensor


def get_1d_rect_bounds(samples, th = 1e-6):
    bounds = {}
    for k, v in samples.items():
        r = v.ratios
        r = r - r.max(axis=0).values  # subtract peak
        p = v.values
        #w = v[..., 0]
        #p = v[..., 1]
        all_max = p.max(dim=0).values
        all_min = p.min(dim=0).values
        constr_min = torch.where(r > np.log(th), p, all_max).min(dim=0).values
        constr_max = torch.where(r > np.log(th), p, all_min).max(dim=0).values
        #bound = torch.stack([constr_min, constr_max], dim = -1)
        bound = RectangleBound(constr_min, constr_max)
        bounds[k] = bound
    return bounds

