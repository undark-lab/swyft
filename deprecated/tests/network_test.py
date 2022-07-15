from itertools import product

import pytest
import torch

from swyft.networks.standardization import OnlineStandardizingLayer


class TestStandardizationLayer:
    bss = [1, 128]
    shapes = [(1,), (5,), (10, 5, 2, 1), (1, 1, 1)]
    means = [2, 139]
    stds = [1, 80]
    stables = [True, False]

    @pytest.mark.parametrize(
        "bs, shape, mean, std, stable", product(bss, shapes, means, stds, stables)
    )
    def test_online_standardization_layer_update(self, bs, shape, mean, std, stable):
        osl = OnlineStandardizingLayer(shape, stable=stable)
        osl.train()
        old_stats = osl.n.clone(), osl.mean.clone(), osl.var.clone(), osl.std.clone()

        data = torch.randn(bs, *shape) * std + mean
        _ = osl(data)

        new_stats = osl.n.clone(), osl._mean.clone(), osl.var.clone(), osl.std.clone()

        assert old_stats != new_stats

    @pytest.mark.parametrize(
        "bs, shape, mean, std, stable", product(bss, shapes, means, stds, stables)
    )
    def test_online_standardization_layer_mean(self, bs, shape, mean, std, stable):
        torch.manual_seed(0)

        osl = OnlineStandardizingLayer(shape, stable=stable)
        osl.train()

        data = torch.randn(bs, *shape) * std + mean
        _ = osl(data)

        assert torch.allclose(osl.mean, data.mean(0))

    @pytest.mark.parametrize(
        "bs, shape, mean, std, stable", product(bss, shapes, means, stds, stables)
    )
    def test_online_standardization_layer_std(self, bs, shape, mean, std, stable):
        torch.manual_seed(0)

        osl = OnlineStandardizingLayer(shape, stable=stable)
        osl.train()

        data = torch.randn(bs, *shape) * std + mean
        _ = osl(data)

        if torch.isnan(data.std(0)).all():
            replacement_std = torch.sqrt(torch.ones_like(osl.std) * osl.epsilon)
            assert torch.allclose(osl.std, replacement_std)
        else:
            assert torch.allclose(osl.std, data.std(0))

    @pytest.mark.parametrize(
        "bs, shape, mean, std, stable", product(bss, shapes, means, stds, stables)
    )
    def test_online_standardization_layer_std_average(
        self, bs, shape, mean, std, stable
    ):
        torch.manual_seed(0)

        osl = OnlineStandardizingLayer(shape, stable=stable, use_average_std=True)
        osl.train()

        data = torch.randn(bs, *shape) * std + mean
        _ = osl(data)

        if torch.isnan(data.std(0)).all():
            replacement_std = torch.sqrt(torch.ones_like(osl.std) * osl.epsilon)
            assert torch.allclose(osl.std, replacement_std.mean())
        else:
            assert torch.allclose(osl.std, data.std(0).mean())


if __name__ == "__main__":
    pass
