from typing import Optional, Union

import numpy as np
import pandas as pd

from swyft.types import Array
from swyft.utils import tensor_to_array


def get_params_weights_from_weighted_marginals(
    weighted_marginals: dict, param_inds: Union[int, tuple]
) -> tuple:
    assert "params" in weighted_marginals
    assert "weights" in weighted_marginals
    if isinstance(param_inds, int):
        param_inds = (param_inds,)
    params = weighted_marginals["params"][:, param_inds]
    weights = weighted_marginals["weights"][param_inds]
    return params, weights


def convert_params_weights_to_df(
    params: Array, weights: Array, param_columns: Optional[tuple] = None
) -> pd.DataFrame:
    params = tensor_to_array(params)
    weights = tensor_to_array(weights)[..., np.newaxis]
    data = np.concatenate([params, weights], axis=-1)

    if isinstance(param_columns, int):
        param_columns = [param_columns]
    elif param_columns is None:
        param_columns = list(range(params.shape[-1]))
    else:
        param_columns = list(param_columns)
    columns = param_columns + ["weights"]
    return pd.DataFrame(data, columns=columns)


def get_df_from_weighted_marginals(
    weighted_marginals: dict, param_inds: Union[int, tuple]
) -> pd.DataFrame:
    return convert_params_weights_to_df(
        *get_params_weights_from_weighted_marginals(weighted_marginals, param_inds),
        param_columns=param_inds,
    )


def get_df_dict_from_weighted_marginals(weighted_marginals: dict) -> dict:
    assert "params" in weighted_marginals
    assert "weights" in weighted_marginals

    dfs = {}
    for key in weighted_marginals["weights"]:
        dfs[key] = get_df_from_weighted_marginals(weighted_marginals, key)
    return dfs
