from dataclasses import dataclass
from typing import Tuple, TypeVar

import numpy as np
import pandas as pd

from swyft.types import (
    MarginalIndex,
    MarginalToArray,
    MarginalToDataFrame,
    StrictMarginalIndex,
)
from swyft.utils.marginals import filter_marginals_by_dim, tupleize_marginal_indices

WeightedMarginalSamplesType = TypeVar(
    "WeightedMarginalSamplesType", bound="WeightedMarginalSamples"
)


@dataclass
class WeightedMarginalSamples:
    weights: MarginalToArray
    v: np.ndarray

    @staticmethod
    def _select_marginal_index(marginal_index: MarginalIndex) -> Tuple[int, ...]:
        marginal_index = tupleize_marginal_indices(marginal_index)
        assert (
            len(marginal_index) == 1
        ), "weighted marginal samples can only be recovered one index at a time"
        return marginal_index[0]

    def get_logweight(self, marginal_index: MarginalIndex) -> np.ndarray:
        """access the logweight for a certain marginal by marginal_index

        Args:
            marginal_index: which marginal to select. one at at time.

        Returns:
            logweight
        """
        marginal_index = self._select_marginal_index(marginal_index)
        logweight = self.weights[marginal_index]
        return logweight

    def get_logweight_marginal(
        self, marginal_index: MarginalIndex
    ) -> Tuple[np.ndarray, np.ndarray]:
        """access the logweight and parameter values for a marginal by index

        Args:
            marginal_index: which marginal to select. one at a time.

        Returns:
            logweight, marginal: the logweight and the parameter values
        """
        marginal_index = self._select_marginal_index(marginal_index)
        logweight = self.get_logweight(marginal_index)
        marginal = self.v[:, marginal_index]
        return logweight, marginal

    def get_df(self, marginal_index: MarginalIndex) -> pd.DataFrame:
        """convert a weighted marginal into a dataframe with the marginal_indices, 'weight', and 'logweight' as columns

        Args:
            marginal_index: which marginal to select. one at a time.

        Returns:
            DataFrame with marginal_indices, 'weight', and 'logweight' for columns
        """
        marginal_index = self._select_marginal_index(marginal_index)
        logweight, marginal = self.get_logweight_marginal(marginal_index)
        weight = np.exp(logweight)

        data = np.concatenate(
            [marginal, weight[..., None], logweight[..., None]], axis=-1
        )
        columns = list(marginal_index) + ["weight"] + ["logweight"]
        return pd.DataFrame(data=data, columns=columns)

    def get_df_dict(self) -> MarginalToDataFrame:
        """produce a map from marginal_index to df for all dfs and marginal_indices"""
        return {
            marginal_index: self.get_df(marginal_index)
            for marginal_index in self.marginal_indices
        }

    def filter_by_dim(self, dim: int) -> WeightedMarginalSamplesType:
        weights = filter_marginals_by_dim(self.weights, dim)
        return self.__class__(weights, self.v)

    @property
    def marginal_indices(self) -> StrictMarginalIndex:
        return tuple(self.weights.keys())
