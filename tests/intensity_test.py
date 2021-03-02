# pylint: disable=no-member
import tempfile
from itertools import product

import numpy as np
import pytest
import torch

from swyft.ip3 import Intensity

# from swyft.ip3 import (
#     FactorMask,
#     Mask1d,
#     construct_intervals,
#     get_factor_mask_from_intervals,
# )


# class TestFactorMask:
#     resolution = 100000

#     xmaxes = [2 * np.pi, 4 * np.pi, 6 * np.pi]
#     areas = [np.pi, 2 * np.pi, 3 * np.pi]

#     @pytest.mark.parametrize("xmax, area", zip(xmaxes, areas))
#     def test_get_factor_mask_from_intervals_1dim(self, xmax, area):
#         x = np.linspace(0, xmax, TestFactorMask.resolution)
#         y = np.sin(x)
#         intervals = construct_intervals(x, y)
#         factor_mask = get_factor_mask_from_intervals(intervals)
#         assert np.isclose(
#             area, factor_mask.area(), rtol=1e-4
#         ), f"Truth {area} != factor mask {factor_mask.area()}."

#     fns = [
#         np.sin,
#         np.cos,
#         lambda x: -((x - 0.5) ** 2) + 0.1,
#     ]
#     area_over_4pi = [
#         2 * np.pi,
#         2 * np.pi,
#         0.816228 - 0.183772,
#     ]
#     fn_area = list(zip(fns, area_over_4pi))

#     @pytest.mark.parametrize("fn_area1, fn_area2", product(fn_area, fn_area))
#     def test_get_factor_mask_from_intervals_2dim(self, fn_area1, fn_area2):
#         fn1, area1 = fn_area1
#         fn2, area2 = fn_area2

#         x = np.linspace(0, 4 * np.pi, TestFactorMask.resolution)
#         y1 = fn1(x)
#         y2 = fn2(x)
#         area = area1 * area2

#         intervals = [construct_intervals(x, y) for y in (y1, y2)]
#         factor_mask = get_factor_mask_from_intervals(intervals)
#         assert np.isclose(
#             area, factor_mask.area(), rtol=1e-4
#         ), f"Truth {area} != factor mask {factor_mask.area()}."


# class TestConstructIntervals:
#     xmaxs = [4 * np.pi, 4 * np.pi, 1.0, 1.0, 1.0]
#     fns = [
#         np.sin,
#         np.cos,
#         lambda x: (x - 0.5) ** 2 - 0.1,
#         lambda x: -((x - 0.5) ** 2) + 0.1,
#         lambda x: x - 0.5,
#     ]
#     targets = [
#         [[0.0, 3.140650081536511], [6.282556925810732, 9.424463770084952]],
#         [
#             [1.5696966593994006, 4.711603503673621],
#             [7.853510347947842, 10.995417192222062],
#         ],
#         [[0.18371837183718373, 0.8161816181618162]],
#         [[0.18371837183718373, 0.8161816181618162]],
#         [[0.4999499949995, 1.0]],
#     ]

#     @pytest.mark.parametrize("fn, target, xmax", zip(fns, targets, xmaxs))
#     def test_construct_intervals(self, fn, target, xmax):
#         x = np.linspace(0, xmax, 10000)
#         y = fn(x)
#         intervals = construct_intervals(x, y)
#         intervals = np.asarray(intervals)
#         target = np.asarray(target)
#         assert np.allclose(target, intervals)


# class TestIntensityIO:
#     expected_ns = [0, 1, 100]
#     intervals = [
#         [[0.0, 3.140650081536511], [6.282556925810732, 9.424463770084952]],
#         [[0.18371837183718373, 0.8161816181618162]],
#     ]

#     @pytest.mark.parametrize("expected_n, intervals", product(expected_ns, intervals))
#     def test_intensity_save_load(self, expected_n, intervals):
#         factor_mask = get_factor_mask_from_intervals(intervals)
#         intensity = Intensity(expected_n, factor_mask)
#         with tempfile.NamedTemporaryFile() as tf:
#             intensity.save(tf.name)

#             loaded = Intensity.load(tf.name)

#         gather_to_check = lambda x: (x.expected_n, x.area, x.factor_mask.intervals)
#         objs = list(map(gather_to_check, [loaded, intensity]))
#         assert [np.isclose(i, j) for i, j in zip(objs[0], objs[1])]
