import tempfile
from itertools import chain, product
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest
import torch
from scipy import stats
from toolz import compose
from torch.distributions import Normal, Uniform

from swyft.prior import InterpolatedTabulatedDistribution, Prior
from swyft.utils import array_to_tensor, tensor_to_array

dimensions = [1, 2, 5]
uniform_hyperparameters = [(torch.zeros(d), torch.ones(d)) for d in dimensions]
uniform_hyperparameters += [
    (-2 * torch.ones(d), -1 * torch.ones(d)) for d in dimensions
]
uniform_hyperparameters += [(-1 * torch.ones(d), 1 * torch.ones(d)) for d in dimensions]
uniform_hyperparameters += [(1 * torch.ones(d), 2 * torch.ones(d)) for d in dimensions]

normal_hyperparameters = [(torch.zeros(d), torch.ones(d)) for d in dimensions]
normal_hyperparameters += [(5 * torch.ones(d), torch.ones(d)) for d in dimensions]
normal_hyperparameters += [
    (5 * torch.randn(d), 0.1 * torch.ones(d)) for d in dimensions
]


class TestInterpolatedTabulatedDistribution:
    def setup_method(self, method):
        torch.manual_seed(0)
        self.n = 1_000
        self.n_grid_points = 10_000

    @pytest.mark.parametrize(
        "distribution, args",
        chain(
            product([Uniform], uniform_hyperparameters),
            product([Normal], normal_hyperparameters),
        ),
    )
    def test_u(self, distribution: Callable, args: Tuple[torch.Tensor, ...]) -> None:
        distribution = distribution(*args)
        samples = distribution.sample((self.n,))
        hypercube_samples = distribution.cdf(samples).numpy()
        _, n_parameters = samples.shape

        itd = InterpolatedTabulatedDistribution(
            compose(tensor_to_array, distribution.icdf, array_to_tensor),
            n_parameters,
            n_grid_points=self.n_grid_points,
        )

        hypercube_samples_itd = itd.cdf(samples)
        assert np.allclose(
            hypercube_samples, hypercube_samples_itd, atol=1e-4, rtol=5e-3
        )

    @pytest.mark.skip(reason="This is an inaccurate feature and should be removed.")
    @pytest.mark.parametrize(
        "distribution, args",
        chain(
            product([Uniform], uniform_hyperparameters),
            product([Normal], normal_hyperparameters),
        ),
    )
    def test_v(self, distribution: Callable, args: Tuple[torch.Tensor, ...]) -> None:
        distribution = distribution(*args)
        samples = distribution.sample((self.n,))
        hypercube_samples = distribution.cdf(samples).numpy()
        _, n_parameters = samples.shape

        itd = InterpolatedTabulatedDistribution(
            compose(tensor_to_array, distribution.icdf, array_to_tensor),
            n_parameters,
            n_grid_points=self.n_grid_points,
        )

        samples_itd = itd.icdf(hypercube_samples)

        if isinstance(distribution, Uniform):
            assert np.allclose(samples, samples_itd)
        elif isinstance(distribution, Normal):
            assert np.allclose(samples, samples_itd, atol=1e-1, rtol=1e-1)


class TestPrior:
    def setup_method(self, method):
        torch.manual_seed(0)
        self.n = 10_000

    @pytest.mark.parametrize(
        "distribution, args",
        chain(
            product([Uniform], uniform_hyperparameters),
            product([Normal], normal_hyperparameters),
        ),
    )
    def test_u_from_torch_distribution(
        self, distribution: Callable, args: Tuple[torch.Tensor, ...]
    ) -> None:
        distribution = distribution(*args)
        samples = distribution.sample((self.n,))
        hypercube_samples_true = distribution.cdf(samples).numpy()

        prior = Prior.from_torch_distribution(distribution)
        hypercube_samples_esti = prior.cdf(samples)
        assert np.allclose(hypercube_samples_true, hypercube_samples_esti)

    @pytest.mark.parametrize(
        "distribution, args",
        chain(
            product([Uniform], uniform_hyperparameters),
            product([Normal], normal_hyperparameters),
        ),
    )
    def test_v_from_torch_distribution(
        self, distribution: Callable, args: Tuple[torch.Tensor, ...]
    ) -> None:
        distribution = distribution(*args)
        samples_true = distribution.sample((self.n,))
        hypercube_samples = distribution.cdf(samples_true).numpy()

        prior = Prior.from_torch_distribution(distribution)
        samples_esti = prior.icdf(hypercube_samples)

        if isinstance(distribution, Uniform):
            assert np.allclose(samples_true, samples_esti)
        elif isinstance(distribution, Normal):
            assert np.allclose(samples_true, samples_esti, atol=1e-4, rtol=5e-3)


class TestSaveLoadPrior:
    @classmethod
    def setup_class(cls):
        cls.directory = tempfile.TemporaryDirectory()

    @classmethod
    def teardown_class(cls):
        cls.directory.cleanup()

    def test_save_load_init(self):
        normal = stats.norm(0, 1)
        prior = Prior(
            cdf=normal.cdf,
            icdf=normal.ppf,
            log_prob=normal.logpdf,
            n_parameters=1,
        )

        # Saving
        path = Path(self.directory.name) / f"from_init_scipy"
        prior.save(path)

        # Loading
        prior_loaded = Prior.load(path)

        # Testing by cdf
        # (icdf or log_prob would also be fine.)
        samples = np.random.randn(1_000)
        cdf_true = prior.log_prob(samples)
        cdf_esti = prior_loaded.log_prob(samples)
        assert np.allclose(cdf_true, cdf_esti)

    @pytest.mark.parametrize(
        "distribution, args",
        [
            (Uniform, uniform_hyperparameters[0]),
            (Normal, normal_hyperparameters[0]),
        ],
    )
    def test_save_load_from_torch_distribution(self, distribution, args):
        distribution = distribution(*args)
        prior = Prior.from_torch_distribution(distribution)

        # Saving
        path = (
            Path(self.directory.name)
            / f"from_torch_distribution_{distribution.__class__.__name__}"
        )
        prior.save(path)

        # Loading
        prior_loaded = Prior.load(path)

        # Testing by cdf
        # (icdf or log_prob would also be fine.)
        samples = prior.distribution.sample((1_000,))
        cdf_true = prior.cdf(samples)
        cdf_esti = prior_loaded.cdf(samples)
        assert np.allclose(cdf_true, cdf_esti)


if __name__ == "__main__":
    pass
