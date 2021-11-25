from toolz import keyfilter

from swyft.types import Array, MarginalIndex, MarginalToArray, StrictMarginalIndex
from swyft.utils.misc import depth


def tupleize_marginals(marginals: MarginalIndex) -> StrictMarginalIndex:
    """Reformat input marginals into sorted and hashable standard form: tuples of tuples.

    input tuples will be respected as coming from the same marginal.
    lists will assumed to be collections of marginals
    """
    if isinstance(marginals, int):
        out = [marginals]
    elif isinstance(marginals, tuple):
        d = depth(marginals)
        if d == 0:
            raise ValueError("how did this happen?")
        elif d == 1:
            return (marginals,)
        elif d == 2:
            return marginals
        else:
            raise ValueError("marginals can only have two layers of depth, no more.")
    else:
        out = list(marginals)

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
