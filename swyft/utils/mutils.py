from typing import Optional

import pandas as pd
from toolz import keyfilter

from swyft.types import Array, RatiosType
from swyft.utils.array import tensor_to_array


def get_marginal_dim_by_key(key: tuple) -> int:
    return len(key)


def get_marginal_dim_by_value(value: Array) -> int:
    return value.shape[-1]


def filter_marginals_by_dim(ratios: RatiosType, dim: int) -> RatiosType:
    assert all(
        isinstance(k, tuple) for k in ratios.keys()
    ), "This function works on tuples of parameters."
    return keyfilter(lambda x: get_marginal_dim_by_key(x) == dim, ratios)


def get_df_from_params(params: Array, columns: Optional[tuple] = None) -> pd.DataFrame:
    params = tensor_to_array(params)
    if isinstance(columns, int):
        columns = [columns]
    elif columns is None:
        columns = list(range(params.shape[-1]))
    else:
        columns = list(columns)
    return pd.DataFrame(params, columns=columns)


def get_df_dict_from_marginals(marginals: dict) -> dict:
    dfs = {}
    for key in marginals.keys():
        dfs[key] = get_df_from_params(marginals[key], key)
    return dfs
