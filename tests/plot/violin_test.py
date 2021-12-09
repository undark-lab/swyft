import numpy as np
import pytest
from matplotlib.axes import Axes

from swyft.plot.violin import violin
from swyft.utils.marginals import get_corner_marginal_indices


class TestViolin:
    @pytest.mark.parametrize("n_parameters", [2, 3])
    def test_violin(self, n_parameters: int) -> None:
        n_samples = 1_000
        marginal_indices_1d, _ = get_corner_marginal_indices(n_parameters)
        marginals = {
            marginal_index: np.random.rand(n_samples)
            for marginal_index in marginal_indices_1d
        }

        ax = violin(marginals)
        assert isinstance(ax, Axes)
