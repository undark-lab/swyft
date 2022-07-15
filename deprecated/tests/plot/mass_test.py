import numpy as np
import pytest
from matplotlib.figure import Figure

from swyft.plot.mass import empirical_z_score_corner
from swyft.utils.marginals import get_corner_marginal_indices


class TestMass:
    @pytest.mark.parametrize("n_parameters", [2, 3])
    def test_empirical_z_score_corner(self, n_parameters: int) -> None:
        n_samples = 1_000
        marginal_indices_1d, marginal_indices_2d = get_corner_marginal_indices(
            n_parameters
        )
        empirical_mass_1d = {
            marginal_index: np.random.rand(n_samples)
            for marginal_index in marginal_indices_1d
        }
        empirical_mass_2d = {
            marginal_index: np.random.rand(n_samples)
            for marginal_index in marginal_indices_2d
        }

        fig, axes = empirical_z_score_corner(empirical_mass_1d, empirical_mass_2d)
        assert isinstance(fig, Figure)
        assert axes.shape == (n_parameters, n_parameters)
