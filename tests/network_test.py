# pylint: disable=no-member, undefined-variable
from itertools import product

import pytest
import torch

from swyft.networks.normalization import OnlineNormalizationLayer


class TestNormalizationLayer:
    bss = [1, 128]
    shapes = [(1,), (5,), (10, 5, 2, 1), (1, 1, 1)]
    means = [2, 139]
    stds = [1, 80]
    stables = [True, False]

    @pytest.mark.parametrize(
        "bs, shape, mean, std, stable", product(bss, shapes, means, stds, stables)
    )
    def test_online_normalization_layer_update(self, bs, shape, mean, std, stable):
        onl = OnlineNormalizationLayer(shape, stable=stable)
        onl.train()
        old_stats = onl.n.clone(), onl.mean.clone(), onl.var.clone(), onl.std.clone()

        data = torch.randn(bs, *shape) * std + mean
        _ = onl(data)

        new_stats = onl.n.clone(), onl._mean.clone(), onl.var.clone(), onl.std.clone()

        assert old_stats != new_stats

    @pytest.mark.parametrize(
        "bs, shape, mean, std, stable", product(bss, shapes, means, stds, stables)
    )
    def test_online_normalization_layer_mean(self, bs, shape, mean, std, stable):
        torch.manual_seed(0)

        onl = OnlineNormalizationLayer(shape, stable=stable)
        onl.train()

        data = torch.randn(bs, *shape) * std + mean
        _ = onl(data)

        assert torch.allclose(onl.mean, data.mean(0))

    @pytest.mark.parametrize(
        "bs, shape, mean, std, stable", product(bss, shapes, means, stds, stables)
    )
    def test_online_normalization_layer_std(self, bs, shape, mean, std, stable):
        torch.manual_seed(0)

        onl = OnlineNormalizationLayer(shape, stable=stable)
        onl.train()

        data = torch.randn(bs, *shape) * std + mean
        _ = onl(data)

        if torch.isnan(data.std(0)).all():
            replacement_std = torch.sqrt(torch.ones_like(onl.std) * onl.epsilon)
            assert torch.allclose(onl.std, replacement_std)
        else:
            assert torch.allclose(onl.std, data.std(0))

    @pytest.mark.parametrize(
        "bs, shape, mean, std, stable", product(bss, shapes, means, stds, stables)
    )
    def test_online_normalization_layer_std_average(self, bs, shape, mean, std, stable):
        torch.manual_seed(0)

        onl = OnlineNormalizationLayer(shape, stable=stable, use_average_std=True)
        onl.train()

        data = torch.randn(bs, *shape) * std + mean
        _ = onl(data)

        if torch.isnan(data.std(0)).all():
            replacement_std = torch.sqrt(torch.ones_like(onl.std) * onl.epsilon)
            assert torch.allclose(onl.std, replacement_std.mean())
        else:
            assert torch.allclose(onl.std, data.std(0).mean())


if __name__ == "__main__":
    pass
