import pytest
from functools import partial
from itertools import product
import tempfile

import numpy as np

from swyft.estimation import Points, RatioEstimator
from swyft import MemoryCache, get_unit_intensity


def sim_repeat_noise(theta, num_copies):
    noise = np.random.randn(num_copies, *theta.shape)
    expanded_theta = np.expand_dims(theta, axis=0)
    return expanded_theta + noise


def setup_points():
        zdim = 10
        num_copies = 6
        xshape = (num_copies, zdim)
        expected_n = 100
        simulator = partial(sim_repeat_noise, num_copies=num_copies)

        cache = MemoryCache(zdim, xshape)
        intensity = get_unit_intensity(expected_n, zdim)
        cache.grow(intensity)
        cache.simulate(simulator)
        return cache, Points(cache, intensity)


class TestPoints:
    def test_points_save_load(self):
        cache, points = setup_points()
        with tempfile.NamedTemporaryFile() as tf:
            points.save(tf.name)

            loaded = Points.load(cache, tf.name)

        gather_to_check = lambda x: (
            x.indices,
            x.xshape,
            x.zdim,
            x.intensity.expected_n,
            x.intensity.area,
            x.intensity.factor_mask.intervals,
        )
        gathered = list(map(gather_to_check, [loaded, points]))
        assert [np.allclose(i, j) for i, j in zip(gathered[0], gathered[1])]


class TestRatioEstimator:
    def test_ratio_estimator_save_load(self):
        cache, points = setup_points()
        re = RatioEstimator(points)
        with tempfile.NamedTemporaryFile() as tf:
            re.save(tf.name)

            loaded = RatioEstimator.load(cache, tf.name)
        
        gather_to_check = lambda x: (
            x.combinations,
            # TODO
            x.points.indices,
            x.points.xshape,
            x.points.zdim,
            x.points.intensity.expected_n,
            x.points.intensity.area,
            x.points.intensity.factor_mask.intervals,
        )
        gathered = list(map(gather_to_check, [loaded, points]))
        assert [np.allclose(i, j) for i, j in zip(gathered[0], gathered[1])]

if __name__ == "__main__":
    pass
