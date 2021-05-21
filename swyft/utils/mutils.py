from toolz import keyfilter

from swyft.types import Array, Marginals


def get_marginal_dim_by_key(key: tuple) -> int:
    return len(key)


def get_marginal_dim_by_value(value: Array) -> int:
    return value.shape[-1]


def filter_marginals_by_dim(marginals: Marginals, dim: int) -> Marginals:
    assert all(
        isinstance(k, tuple) for k in marginals.keys()
    ), "This function works on tuples of parameters."
    return keyfilter(lambda x: get_marginal_dim_by_key(x) == dim, marginals)
