import pytest

import numpy as np

from swyft.estimation import RatioEstimator
from swyft.eval import eval_net, get_ratios

from .estimation_test import sim_repeat_noise, setup_points, Head


class TestEvaluation:
    @staticmethod
    def setup_re_x0_z0_points():
        _, points = setup_points()
        head = Head(*Head.featurize(points.xshape))
        re = RatioEstimator(points, head=head)

        z0 = np.random.rand(points.zdim)
        x0 = sim_repeat_noise(z0, points.xshape[0])
        z0 = z0[None, :, None]
        x0 = x0
        return re, x0, z0, points

    @pytest.mark.parametrize("training", (True, False))
    def test_eval_net_network_state(self, training):
        re, x0, z0, _ = TestEvaluation.setup_re_x0_z0_points()

        if training:
            re.net.train()
        else:
            re.net.eval()

        _ = eval_net(x0, re.net, z0, batch_size=1)

        assert re.net.training == training

    @pytest.mark.parametrize("training", (True, False))
    def test_get_ratios_network_state(self, training):
        re, x0, _, points = TestEvaluation.setup_re_x0_z0_points()

        if training:
            re.net.train()
        else:
            re.net.eval()

        _ = get_ratios(x0, re.net, points)

        assert re.net.training == training
