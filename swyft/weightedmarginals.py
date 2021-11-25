from dataclasses import dataclass
from typing import Tuple

import numpy as np

from swyft.types import MarginalIndex, MarginalToArray, StrictMarginalIndex
from swyft.utils.marginals import tupleize_marginals


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
