import tempfile
import numpy as np
import pytest
import time
from pathlib import Path
from itertools import product
from swyft.cache.cache import Cache, DirectoryCache, MemoryCache
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


if __name__ == "__main__":
    pass
