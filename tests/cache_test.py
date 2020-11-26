from itertools import product

import pytest
import glob
import tempfile

import torch
from swyft.cache import Cache, MemoryCache, DirectoryCache


class TestCacheIO:
    zdims = [0, 1, 5]
    xshapes = [(0,), (1,), (4, 6, 2, 1)]

    @pytest.mark.parametrize("zdim, xshape", product(zdims, xshapes))
    def test_memory_cache_save(self, zdim, xshape):
        cache = MemoryCache(zdim, xshape)
        with tempfile.TemporaryDirectory() as td:
            cache.save(td)
            items = glob.glob(td + "/**", recursive=True)

        without_prefix = sorted([item[len(td) + 1 :] for item in items])
        without_blanks = sorted([item for item in without_prefix if item])
        assert all(
            [item == truth for item, truth in zip(without_blanks, Cache._filesystem)]
        )

    @pytest.mark.parametrize("zdim, xshape", product(zdims, xshapes))
    def test_memory_cache_load(self, zdim, xshape):
        cache = MemoryCache(zdim, xshape)
        cache_states = (cache.zdim, cache.xshape, len(cache))

        with tempfile.TemporaryDirectory() as td:
            cache.save(td)
            loaded = MemoryCache.load(td)
            loaded_stats = (loaded.zdim, loaded.xshape, len(loaded))
            assert cache_states == loaded_stats

    @pytest.mark.parametrize("zdim, xshape", product(zdims, xshapes))
    def test_directory_cache_load(self, zdim, xshape):
        cache = MemoryCache(zdim, xshape)
        cache_states = (cache.zdim, cache.xshape, len(cache))

        with tempfile.TemporaryDirectory() as td:
            cache.save(td)
            loaded = DirectoryCache.load(td)
            loaded_stats = (loaded.zdim, loaded.xshape, len(loaded))
            assert cache_states == loaded_stats


if __name__ == "__main__":
    pass
