import tempfile
from functools import partial

import numpy as np
import pytest
import torch
import torch.nn as nn

from swyft.cache import MemoryCache
from swyft.inference import RatioEstimator

# from swyft.ip3 import Points, get_unit_intensity


# def sim_repeat_noise(theta, num_copies):
#     noise = np.random.randn(num_copies, *theta.shape)
#     expanded_theta = np.expand_dims(theta, axis=0)
#     return expanded_theta + noise


# def setup_points():
#     zdim = 10
#     num_copies = 3
#     xshape = (num_copies, zdim)
#     expected_n = 100
#     simulator = partial(sim_repeat_noise, num_copies=num_copies)

#     cache = MemoryCache(zdim, xshape)
#     intensity = get_unit_intensity(expected_n, zdim)
#     cache.grow(intensity)
#     cache.simulate(simulator)
#     return cache, Points(cache, intensity)


# class Head(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.flatten = torch.nn.Flatten()
#         self.layer = torch.nn.Linear(in_features, out_features)

#     def forward(self, x):
#         return self.layer(self.flatten(x))

#     @staticmethod
#     def featurize(xshape):
#         in_features = np.product(xshape)
#         out_features = 100
#         return in_features, out_features


# class TestPoints:
#     def test_points_save_load(self):
#         cache, points = setup_points()
#         with tempfile.NamedTemporaryFile() as tf:
#             points.save(tf.name)

#             loaded = Points.load(cache, tf.name)

#         gather_attrs = lambda x: (
#             x.indices,
#             x.xshape,
#             x.zdim,
#             x.intensity.expected_n,
#             x.intensity.area,
#             x.intensity.factor_mask.intervals,
#         )
#         assert [
#             np.allclose(i, j)
#             for i, j in zip(gather_attrs(loaded), gather_attrs(points))
#         ]


# class TestRatioEstimator:
#     @pytest.mark.parametrize("head", (None, Head))
#     def test_ratio_estimator_save_load(self, head):
#         cache, points = setup_points()

#         if head is None:
#             head1 = None
#             head2 = None
#         else:
#             in_features, out_features = Head.featurize(points.xshape)
#             head1 = head(in_features, out_features)
#             head2 = head(in_features, out_features)

#         re = RatioEstimator(points, head=head1)
#         with tempfile.NamedTemporaryFile() as tf:
#             re.save(tf.name)

#             loaded = RatioEstimator.load(cache, tf.name, head=head2)

#         gather_attrs = lambda x: (
#             x.combinations,
#             *(v for _, v in x.net_state_dict.items()),
#             *(v for _, v in x.ratio_cache.items()),
#             x.points.indices,
#             x.points.xshape,
#             x.points.zdim,
#             x.points.intensity.expected_n,
#             x.points.intensity.area,
#             x.points.intensity.factor_mask.intervals,
#         )
#         assert [
#             np.allclose(i, j) for i, j in zip(gather_attrs(loaded), gather_attrs(re))
#         ]


if __name__ == "__main__":
    pass
