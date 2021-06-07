import tempfile
import time
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from swyft import Prior
from swyft.cache.cache import Cache, DirectoryCache, MemoryCache
from swyft.utils.simulator import Simulator


def model(params):
    p = np.linspace(-1, 1, 10)  # Nbin = 10
    mu = params["a"] + p * params["b"]
    return dict(x=mu)


prior = Prior({"a": ["uniform", 0.0, 1.0], "b": ["uniform", 0.0, 1.0]})


class TestCacheIO:
    def test_init_memory_cache(self):
        cache = MemoryCache(["a", "b", "c"], {"x": (10,)})
        assert cache.params == ["a", "b", "c"]
        assert {
            k: v.shape[1:] for k, v in cache.root[Cache._filesystem.obs].items()
        } == {"x": (10,)}

    def test_init_memory_cache_multi_outputs(self):
        cache = MemoryCache(["a", "b"], {"x1": (10,), "x2": (20, 20)})
        assert cache.params == ["a", "b"]
        assert {
            k: v.shape[1:] for k, v in cache.root[Cache._filesystem.obs].items()
        } == {"x1": (10,), "x2": (20, 20)}

    def test_init_directory_cache_multi_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            cache = DirectoryCache(["a", "b"], {"x1": (10,), "x2": (20, 20)}, path=td)
            assert cache.params == ["a", "b"]
            assert {
                k: v.shape[1:] for k, v in cache.root[Cache._filesystem.obs].items()
            } == {"x1": (10,), "x2": (20, 20)}

            td_path = Path(td)
            items = [
                p.relative_to(td).as_posix() for p in td_path.rglob("*") if p.is_dir()
            ]
            assert all([truth in items for truth in Cache._filesystem])

    def test_memory_cache_save(self):
        cache = MemoryCache.from_simulator(model, prior)
        with tempfile.TemporaryDirectory() as td:
            cache.save(td)
            td_path = Path(td)
            items = [
                p.relative_to(td).as_posix() for p in td_path.rglob("*") if p.is_dir()
            ]
            assert all([truth in items for truth in Cache._filesystem])

    def test_memory_cache_load(self):
        cache = MemoryCache.from_simulator(model, prior)
        with tempfile.TemporaryDirectory() as td:
            cache.save(td)
            loaded = MemoryCache.load(td)
            assert loaded.params == cache.params
            assert cache.store.root == loaded.store.root

    def test_directory_cache_load(self):
        with tempfile.TemporaryDirectory() as td:
            cache = DirectoryCache(["a", "b"], {"x": (10,)}, path=td, sync_path=td)
            loaded = DirectoryCache.load(td)
            obs_shape_cache = {
                k: v.shape[1:] for k, v in cache.root[Cache._filesystem.obs].items()
            }
            obs_shape_loaded = {
                k: v.shape[1:] for k, v in cache.root[Cache._filesystem.obs].items()
            }
            assert loaded.params == cache.params
            assert obs_shape_cache == obs_shape_loaded


class TestCacheRun:
    def test_memory_cache_sample(self):
        cache = MemoryCache.from_simulator(model, prior)
        indices = cache.sample(prior, 1000)
        assert len(indices) == len(cache)

    def test_memory_cache_simulate(self):
        simulator = Simulator(model)
        cache = MemoryCache.from_simulator(model, prior)
        indices = cache.sample(prior, 200)
        ind_sim = indices[:50]
        cache.simulate(simulator, ind_sim)

        assert cache.x["x"][49].sum() != 0
        assert cache.x["x"][50].sum() == 0

    def test_directory_cache_sample(self):
        with tempfile.TemporaryDirectory() as td:
            cache = DirectoryCache(["a", "b"], {"x": (10,)}, path=td)
            indices = cache.sample(prior, 1000)
            assert len(indices) == len(cache)

    def test_directory_cache_simulate(self):
        simulator = Simulator(model)
        with tempfile.TemporaryDirectory() as td:
            cache = DirectoryCache(["a", "b"], {"x": (10,)}, path=td)
            indices = cache.sample(prior, 200)
            cache.simulate(simulator, indices)
            assert cache.x["x"][:].sum(axis=1).all()

    def test_directory_cache_simulate_partial(self):
        simulator = Simulator(model)
        with tempfile.TemporaryDirectory() as td:
            cache = DirectoryCache(["a", "b"], {"x": (10,)}, path=td)
            indices = cache.sample(prior, 200)
            ind_sim = indices[:50]
            cache.simulate(simulator, ind_sim)
            assert cache.x["x"][49].sum() != 0
            assert cache.x["x"][50].sum() == 0

    def test_cache_lockfile(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = DirectoryCache(
                ["a", "b"], {"x": (10,)}, path=td, sync_path=td + ".sync"
            )
            assert cache_dir._lock is not None
            assert cache_dir._lock.lockfile is None

            cache_dir.lock()
            assert cache_dir._lock.lockfile is not None

            cache_dir.unlock()
            assert cache_dir._lock.lockfile is None


if __name__ == "__main__":
    pass
