import numpy as np
import pytest
from matplotlib.figure import Figure

from swyft.plot.histogram import corner, hist1d
from swyft.utils.marginals import filter_marginals_by_dim, get_corner_marginal_indices
from swyft.weightedmarginals import WeightedMarginalSamples


class TestCorner:
    @pytest.mark.parametrize("n_parameters", [2, 3])
    def test_corner_weighted_samples(self, n_parameters: int) -> None:
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

        weighted_samples = WeightedMarginalSamples(weights, v)
        fig, axes = corner(
            marginal_1d=weighted_samples.filter_by_dim(1),
            marginal_2d=weighted_samples.filter_by_dim(2),
        )
        assert isinstance(fig, Figure)
        assert axes.shape == (n_parameters, n_parameters)

    @pytest.mark.parametrize("n_parameters", [2, 3])
    def test_corner_samples(self, n_parameters: int) -> None:
        n_samples = 1_000
        marginal_indices_1d, marginal_indices_2d = get_corner_marginal_indices(
            n_parameters
        )
        marginal_indices = list(marginal_indices_1d) + list(marginal_indices_2d)
        marginal_samples = {
            marginal_index: np.random.rand(n_samples, len(marginal_index))
            for marginal_index in marginal_indices
        }

        fig, axes = corner(
            marginal_1d=filter_marginals_by_dim(marginal_samples, 1),
            marginal_2d=filter_marginals_by_dim(marginal_samples, 2),
        )
        assert isinstance(fig, Figure)
        assert axes.shape == (n_parameters, n_parameters)


class TestHist1d:
    @pytest.mark.parametrize("n_parameters", [2, 3])
    def test_hist1d_weighted_samples(self, n_parameters: int) -> None:
        n_samples = 1_000
        marginal_indices, _ = get_corner_marginal_indices(n_parameters)
        v = 0.1 * np.random.rand(n_samples, n_parameters) + 0.45
        weights = {
            marginal_index: np.random.rand(n_samples)
            for marginal_index in marginal_indices
        }

        weighted_samples = WeightedMarginalSamples(weights, v)
        fig, axes = hist1d(
            marginal_1d=weighted_samples,
        )
        assert isinstance(fig, Figure)
        assert axes.shape == (n_parameters,)

    @pytest.mark.parametrize("n_parameters", [2, 3])
    def test_hist1d_samples(self, n_parameters: int) -> None:
        n_samples = 1_000
        marginal_indices, _ = get_corner_marginal_indices(n_parameters)
        marginal_samples = {
            marginal_index: np.random.rand(n_samples, len(marginal_index))
            for marginal_index in marginal_indices
        }

        fig, axes = hist1d(
            marginal_1d=marginal_samples,
        )
        assert isinstance(fig, Figure)
        assert axes.shape == (n_parameters,)
