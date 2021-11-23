from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from toolz.dicttoolz import keyfilter

from swyft.types import Array, MarginalIndex, MarginalToArray, StrictMarginalIndex
from swyft.utils import tensor_to_array
from swyft.utils.marginalutils import tupleize_marginals


@dataclass
class WeightedMarginalSamples:
    weights: MarginalToArray
    v: np.ndarray

    @staticmethod
    def _select_marginal_index(marginal_index: MarginalIndex) -> Tuple[int, ...]:
        marginal_index = tupleize_marginals(marginal_index)
        assert (
            len(marginal_index) == 1
        ), "weighted marginal samples can only be recovered one index at a time"
        return marginal_index[0]

    def get_logweight(self, marginal_index: MarginalIndex):
        marginal_index = self._select_marginal_index(marginal_index)
        log_weight = self.weights[marginal_index]
        return log_weight

    def get_logweight_marginal(
        self, marginal_index: MarginalIndex
    ) -> Tuple[np.ndarray, np.ndarray]:
        """access the log_weight and parameter values for a marginal by index

        Args:
            marginal_index: which marginal to select. one at a time.

        Returns:
            log_weights, marginal: the log_weights and the parameter values
        """
        marginal_index = self._select_marginal_index(marginal_index)
        log_weight = self.get_logweight(marginal_index)
        marginal = self.v[:, marginal_index]
        return log_weight, marginal

    @property
    def marginal_indices(self) -> StrictMarginalIndex:
        return tuple(self.weights.keys())


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
