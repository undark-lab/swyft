import tempfile
import numpy as np
import pytest
import time
from pathlib import Path
from itertools import product
from swyft.cache.cache import Cache, DirectoryCache, MemoryCache
from swyft.utils.simulator import Simulator
from swyft import Prior


def model(params):
    p = np.linspace(-1, 1, 10)  # Nbin = 10
    mu = params['a'] + p*params['b']
    return dict(x=mu)


prior = Prior({"a": ['uniform', 0., 1.], "b": ['uniform', 0., 1.]})


class TestCacheIO:
    def test_memory_cache_save(self):
        cache = MemoryCache.from_simulator(model, prior)
        with tempfile.TemporaryDirectory() as td:
            cache.save(td)
            td_path = Path(td)
            items = [p.relative_to(td).as_posix()
                     for p in td_path.rglob("*") if p.is_dir()]
        assert all([truth in items for truth in Cache._filesystem])

    def test_memory_cache_load(self):
        cache = MemoryCache.from_simulator(model, prior)
        with tempfile.TemporaryDirectory() as td:
            cache.save(td)
            loaded = MemoryCache.load(td)

            assert loaded.params == cache.params
            assert cache.store.root == loaded.store.root


class TestCacheRun:
    def test_cache_sample(self):
        cache = MemoryCache.from_simulator(model, prior)
        indices = cache.sample(prior, 1000)
        assert len(indices) == len(cache)

    def test_cache_simulate(self):
        simulator = Simulator(model)
        cache = MemoryCache.from_simulator(model, prior)
        indices = cache.sample(prior, 200)
        ind_sim = indices[:50]
        cache.simulate(simulator, ind_sim)

        assert cache.x['x'][49].sum() != 0
        assert cache.x['x'][50].sum() == 0

    def test_cache_lock_and_unlock(self):
        with tempfile.TemporaryDirectory() as td:
            cache_mem = MemoryCache.from_simulator(model, prior)
            cache_mem.save(td)
            cache_dir = DirectoryCache(['a', 'b'], (10,), path=td, sync_path=td)
            assert cache_dir._lock is not None
            assert cache_dir._lock.lockfile is None

            cache_dir.lock()
            assert cache_dir._lock.lockfile is not None

            cache_dir.unlock()
            assert cache_dir._lock.lockfile is None


if __name__ == "__main__":
    pass
