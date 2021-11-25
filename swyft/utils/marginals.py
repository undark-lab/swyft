from itertools import combinations
from typing import List, Tuple

from toolz import keyfilter

from swyft.types import Array, MarginalIndex, MarginalToArray, StrictMarginalIndex
from swyft.utils.misc import depth


def get_all_d_dim_marginals(n_parameters: int, d: int) -> StrictMarginalIndex:
    return tuple(combinations(range(n_parameters), d))


def get_corner_marginals(
    n_parameters: int,
) -> Tuple[StrictMarginalIndex, StrictMarginalIndex]:
    """produce the marginals for a corner plot

    Args:
        n_parameters

    Returns:
        marginal_indices_1d, marginal_indices_2d
    """
    marginal_indices_1d = get_all_d_dim_marginals(n_parameters, 1)
    marginal_indices_2d = get_all_d_dim_marginals(n_parameters, 2)
    return marginal_indices_1d, marginal_indices_2d


def tupleize_marginal_indices(marginal_indices: MarginalIndex) -> StrictMarginalIndex:
    """Reformat input marginal_indices into sorted and hashable standard form: tuples of tuples.

    a lone input tuple will be respected as coming from the same marginal
    lists will assumed to be collections of different marginals
    """
    if isinstance(marginal_indices, int):
        out = [marginal_indices]
    elif isinstance(marginal_indices, tuple):
        d = depth(marginal_indices)
        if d == 0:
            raise ValueError("how did this happen?")
        elif d == 1:
            return (marginal_indices,)
        elif d == 2:
            return marginal_indices
        else:
            raise ValueError("marginals can only have two layers of depth, no more.")
    else:
        out = list(marginal_indices)

    for i in range(len(out)):
        if isinstance(out[i], int):
            out[i] = (out[i],)
        else:
            out[i] = tuple(sorted(set(out[i])))
    out = tuple(sorted(out))
    return out


def get_marginal_dim_by_key(key: tuple) -> int:
    return len(key)


def get_marginal_dim_by_value(value: Array) -> int:
    return value.shape[-1]


def filter_marginals_by_dim(ratios: MarginalToArray, dim: int) -> MarginalToArray:
    assert all(
        isinstance(k, tuple) for k in ratios.keys()
    ), "This function works on tuples of parameters."
    return keyfilter(lambda x: get_marginal_dim_by_key(x) == dim, ratios)
