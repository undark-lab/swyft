import glob
import tempfile
from itertools import product

import pytest

from swyft.cache.cache import Cache, DirectoryCache, MemoryCache

# class TestCacheIO:
#     params = ["one", "two", "three"]
#     obs_shapes = [(0,), (1,), (4, 6, 2, 1)]

#     @pytest.mark.parametrize("params, obs_shapes", product(params, obs_shapes))
#     def test_memory_cache_save(self, params, obs_shapes):
#         cache = MemoryCache(params, obs_shapes)
#         with tempfile.TemporaryDirectory() as td:
#             cache.save(td)
#             items = glob.glob(td + "/**", recursive=True)

#         without_prefix = sorted([item[len(td) + 1 :] for item in items])
#         without_blanks = sorted([item for item in without_prefix if item])
#         assert all(
#             [item == truth for item, truth in zip(without_blanks, Cache._filesystem)]
#         )

#     @pytest.mark.parametrize("params, obs_shapes", product(params, obs_shapes))
#     def test_memory_cache_load(self, params, obs_shapes):
#         cache = MemoryCache(params, obs_shapes)
#         cache_states = (cache.zdim, cache.xshape, len(cache))

#         with tempfile.TemporaryDirectory() as td:
#             cache.save(td)
#             loaded = MemoryCache.load(td)
#             loaded_stats = (loaded.zdim, loaded.xshape, len(loaded))
#             assert cache_states == loaded_stats

#     @pytest.mark.parametrize("params, obs_shapes", product(params, obs_shapes))
#     def test_directory_cache_load(self, params, obs_shapes):
#         cache = MemoryCache(params, obs_shapes)
#         cache_states = (cache.zdim, cache.xshape, len(cache))

#         with tempfile.TemporaryDirectory() as td:
#             cache.save(td)
#             loaded = DirectoryCache.load(td)
#             loaded_stats = (loaded.zdim, loaded.xshape, len(loaded))
#             assert cache_states == loaded_stats


if __name__ == "__main__":
    pass
