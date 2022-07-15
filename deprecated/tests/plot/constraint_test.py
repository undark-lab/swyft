import numpy as np
import pytest

from swyft.plot.constraint import diagonal_constraint, lower_constraint
from swyft.plot.histogram import corner
from swyft.utils.marginals import get_corner_marginal_indices
from swyft.weightedmarginals import WeightedMarginalSamples


class TestConstraint:
    @pytest.mark.skip
    def test_constraint(self):
        n_samples = 1_000
        n_parameters = 3

        marginal_indices_1d, marginal_indices_2d = get_corner_marginal_indices(
            n_parameters
        )
        marginal_indices = list(marginal_indices_1d) + list(marginal_indices_2d)
        v = 0.1 * np.random.rand(n_samples, n_parameters) + 0.45
        weights = {
            marginal_index: np.random.rand(n_samples)
            for marginal_index in marginal_indices
        }
        df_dict = WeightedMarginalSamples(weights, v).get_df_dict()

        bounds = np.asarray([[0.1, 0.9], [0.2, 0.8], [0.3, 0.9]])
        boundss = [bounds, [1.2, 0.8] * bounds]

        _, axes = corner(df_dict)
        for bounds in boundss:
            diagonal_constraint(axes, bounds)
        for bounds in boundss:
            lower_constraint(axes, bounds)
