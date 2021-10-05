from itertools import product, chain
from typing import Callable, Tuple
from toolz import compose

import pytest
import torch
import numpy as np
from torch.distributions import Uniform, Normal

from swyft.prior import InterpolatedTabulatedDistribution
from swyft.utils import tensor_to_array
from swyft.utils.array import array_to_tensor


class TestInterpolatedTabulatedDistribution:
    dimensions = [1, 2, 5]
    uniform_hyperparameters = [
        (torch.zeros(d), torch.ones(d)) for d in dimensions
    ]
    uniform_hyperparameters += [
        (-2 * torch.ones(d), -1 * torch.ones(d)) for d in dimensions
    ]
    uniform_hyperparameters += [
        (-1 * torch.ones(d), 1 * torch.ones(d)) for d in dimensions
    ]
    uniform_hyperparameters += [
        (1 * torch.ones(d), 2 * torch.ones(d)) for d in dimensions
    ]

    normal_hyperparameters = [
        (torch.randn(d), torch.ones(d)) for d in dimensions
    ]

    @pytest.mark.parametrize(
        "distribution, args", 
        chain(product([Uniform], uniform_hyperparameters), product([Normal], normal_hyperparameters))
    )
    def test_u(self, distribution: Callable, args: Tuple[torch.Tensor, ...]) -> None:
        distribution = distribution(*args)
        
        n = 1_000
        n_grid_points = 100_000
        samples = distribution.sample((n,))
        hypercube_samples = distribution.cdf(samples).numpy()
        _, n_parameters = samples.shape

        itd = InterpolatedTabulatedDistribution(
            compose(tensor_to_array, distribution.icdf, array_to_tensor),
            n_parameters,
            n_grid_points=n_grid_points,
        )
        print(samples.shape)
        hypercube_samples_itd = itd.u(samples)
        assert np.allclose(hypercube_samples, hypercube_samples_itd, rtol=1e-4)


class TestPrior:
    pass


if __name__ == "__main__":
    pass
