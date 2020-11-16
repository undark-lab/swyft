# pylint: disable=no-member
import pytest

import torch
import numpy as np
from swyft.intensity import (
    construct_intervals,
    get_factor_mask_from_intervals,
    Mask1d,
    FactorMask,
    Intensity,
)


class TestFactorMask:
    xmaxes = [2 * np.pi, 4 * np.pi, 6 * np.pi]
    areas = [np.pi, 2 * np.pi, 3 * np.pi]
    resolution = 100000

    @pytest.mark.parametrize("xmax, area", zip(xmaxes, areas))
    def test_get_factor_mask_from_intervals_1dim(self, xmax, area):
        x = np.linspace(0, xmax, TestFactorMask.resolution)
        y = np.sin(x)
        intervals = construct_intervals(x, y)
        factor_mask = get_factor_mask_from_intervals(intervals)
        assert round(area, 2) == round(factor_mask.area(), 2)

    @pytest.mark.parametrize("xmax, area", zip(xmaxes, areas))
    def test_get_factor_mask_from_intervals_2dim(self, xmax, area):
        x = np.linspace(0, xmax, TestFactorMask.resolution)
        y1 = np.sin(x)
        y2 = np.cos(x)
        intervals = [construct_intervals(x, y) for y in (y1, y2)]
        factor_mask = get_factor_mask_from_intervals(intervals)
        assert np.isclose(round(area ** 2, 2), round(factor_mask.area(), 2), rtol=0.001)


class TestConstructIntervals:
    xmaxs = [4 * np.pi, 4 * np.pi, 1.0, 1.0]
    fns = [
        np.sin,
        np.cos,
        lambda x: (x - 0.5) ** 2 - 0.1,
        lambda x: -((x - 0.5) ** 2) + 0.1,
    ]
    targets = [
        [[0.0, 3.140650081536511], [6.282556925810732, 9.424463770084952]],
        [
            [1.5696966593994006, 4.711603503673621],
            [7.853510347947842, 10.995417192222062],
        ],
        [[0.18371837183718373, 0.8161816181618162]],
        [[0.18371837183718373, 0.8161816181618162]],
    ]

    @pytest.mark.parametrize("fn, target, xmax", zip(fns, targets, xmaxs))
    def test_construct_intervals(self, fn, target, xmax):
        x = np.linspace(0, xmax, 10000)
        y = fn(x)
        intervals = construct_intervals(x, y)
        intervals = np.asarray(intervals)
        target = np.asarray(target)
        assert np.allclose(target, intervals)
