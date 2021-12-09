import numpy as np
import pytest
from matplotlib.figure import Figure

from swyft.plot.histogram import corner
from swyft.utils.marginals import filter_marginals_by_dim, get_corner_marginal_indices
from swyft.weightedmarginals import WeightedMarginalSamples


class TestCorner:
    @pytest.mark.parametrize("n_parameters", [2, 3])
    def test_corner(self, n_parameters: int) -> None:
        n_samples = 1_000
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
        fig, axes = corner(
            marginal_1d=filter_marginals_by_dim(df_dict, 1),
            marginal_2d=filter_marginals_by_dim(df_dict, 2),
        )
        assert isinstance(fig, Figure)
        assert axes.shape == (n_parameters, n_parameters)
