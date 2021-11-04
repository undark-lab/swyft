import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pytest

import swyft
from swyft.bounds import RectangleBound, UnitCubeBound
from swyft.prior import PriorTruncator, get_diagonal_normal_prior, get_uniform_prior


class TestSaveLoadPriorTruncator:
    dimension = 10
    n_samples = 1_000
    rectangle_bounds = np.stack([np.zeros(dimension), np.ones(dimension)], axis=-1)

    distributions = [
        get_uniform_prior(low=-1 * np.ones(dimension), high=2 * np.ones(dimension)),
        get_diagonal_normal_prior(
            loc=np.random.randn(dimension), scale=np.ones(dimension)
        ),
    ]
    bounds = [
        None,
        UnitCubeBound(dimension),
        RectangleBound(rectangle_bounds),
    ]

    @classmethod
    def setup_class(cls):
        cls.directory = tempfile.TemporaryDirectory()

    @classmethod
    def teardown_class(cls):
        cls.directory.cleanup()

    @pytest.mark.parametrize("prior, bound", product(distributions, bounds))
    def test_save_load_prior_truncator_no_bound(
        self, prior: swyft.Prior, bound: swyft.Bound
    ):
        prior_truncator = PriorTruncator(prior, bound=None)

        # Saving
        path = Path(self.directory.name) / f"prior_truncator_{bound}"
        prior_truncator.save(path)

        # Loading
        prior_truncator_loaded = PriorTruncator.load(path)

        # Testing by cdf
        # (icdf or log_prob would also be fine.)
        samples = prior_truncator.sample(self.n_samples)

        log_prob_true = prior_truncator.log_prob(samples)
        log_prob_esti = prior_truncator_loaded.log_prob(samples)
        assert np.allclose(log_prob_true, log_prob_esti)


if __name__ == "__main__":
    pass
