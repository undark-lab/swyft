# from itertools import combinations
# from typing import List, Tuple
#
##from pandas.core.frame import DataFrame
# from toolz import keyfilter
#
from swyft.types import (
    Array,
    MarginalIndex,
    MarginalToArray,
    # MarginalToDataFrame,
    StrictMarginalIndex,
)

# from swyft.utils.array import tensor_to_array
from swyft.utils.misc import depth


# def get_d_dim_marginal_indices(n_parameters: int, d: int) -> StrictMarginalIndex:
#    return tuple(combinations(range(n_parameters), d))


# def get_corner_marginal_indices(
#    n_parameters: int,
# ) -> Tuple[StrictMarginalIndex, StrictMarginalIndex]:
#    """produce the marginals for a corner plot
#
#    Args:
#        n_parameters
#
#    Returns:
#        marginal_indices_1d, marginal_indices_2d
#    """
#    marginal_indices_1d = get_d_dim_marginal_indices(n_parameters, 1)
#    marginal_indices_2d = get_d_dim_marginal_indices(n_parameters, 2)
#    return marginal_indices_1d, marginal_indices_2d


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


# def get_marginal_dim_by_key(key: Tuple[int]) -> int:
#    return len(key)
#
#
# def get_marginal_dim_by_value(value: Array) -> int:
#    return value.shape[-1]
#
#
# def filter_marginals_by_dim(marginals: MarginalToArray, dim: int) -> MarginalToArray:
#    assert all(
#        isinstance(k, tuple) for k in marginals.keys()
#    ), "This function works on tuples of parameters."
#    return keyfilter(lambda x: get_marginal_dim_by_key(x) == dim, marginals)


# def get_df_from_marginal(v: Array, marginal_index: Tuple[int] = None) -> DataFrame:
#    v = tensor_to_array(v)
#    if isinstance(marginal_index, int):
#        marginal_index = [marginal_index]
#    elif marginal_index is None:
#        marginal_index = list(range(v.shape[-1]))
#    else:
#        marginal_index = list(marginal_index)
#    return DataFrame(v, columns=marginal_index)


# def get_df_dict_from_marginals(marginals: MarginalToArray) -> MarginalToDataFrame:
#    return {key: get_df_from_marginal(marginals[key], key) for key in marginals.keys()}
