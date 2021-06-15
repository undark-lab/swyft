import tempfile
import numpy as np
import pytest
import time
import zarr
from pathlib import Path
from itertools import product
from swyft.store.store import DirectoryStore, MemoryStore
from swyft.store.simulator import Simulator
from swyft import Prior


def model(params):
    p = np.linspace(-1, 1, 10)
    a, b = params
    mu = p * a + b
    return dict(x=mu)


def model_multi_out(params):
    p = np.linspace(-1, 1, 10)
    a, b = params
    mu = p * a + b
    mu2 = np.reshape(p * a - b, (2, 5))
    return dict(x1=mu, x2=mu2)


sim = Simulator(model, sim_shapes=dict(x=(10,)))
sim_multi_out = Simulator(model_multi_out, sim_shapes=dict(x1=(10,), x2=(2, 5)))
prior = Prior.from_uv(lambda u: u * np.array([1.0, 0.5]), 2)


class TestStoreIO:
    def test_init_memory_store(self):
        store = MemoryStore(2, simulator=sim)
        assert len(store.params) == 2
        assert isinstance(store.zarr_store, zarr.storage.MemoryStore)
        assert isinstance(store._simulator, Simulator)

    def test_init_memory_store_multi_outputs(self):

        store = MemoryStore(2, simulator=sim_multi_out)
        assert len(store.params) == 2
        assert {k: v for k, v in store._simulator.sim_shapes.items()} == {
            "x1": (10,),
            "x2": (2, 5),
        }

    def test_init_directory_store_multi_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            store = DirectoryStore(2, simulator=sim_multi_out, path=td)
            assert len(store.params) == 2
            assert {k: v for k, v in store._simulator.sim_shapes.items()} == {
                "x1": (10,),
                "x2": (2, 5),
            }
            td_path = Path(td)
            items = [
                p.relative_to(td).as_posix() for p in td_path.rglob("*/") if p.is_dir()
            ]
            assert len(items) > 0

    def test_memory_store_save(self):
        store = MemoryStore.from_model(model, prior)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            store.save(td)
            items = [
                p.relative_to(td).as_posix() for p in td_path.rglob("*") if p.is_dir()
            ]
            assert len(items) > 0

    def test_memory_store_load(self):
        store = MemoryStore(2, simulator=sim_multi_out)
        with tempfile.TemporaryDirectory() as td:
            store.save(td)
            loaded = MemoryStore.load(td)
            loaded.set_simulator(sim_multi_out)
            assert loaded.params == store.params
            assert loaded.zarr_store.root == store.zarr_store.root
            assert loaded._simulator.sim_shapes == sim_multi_out.sim_shapes

    def test_directory_store_load(self):
        with tempfile.TemporaryDirectory() as td:
            store = DirectoryStore(2, simulator=sim_multi_out, path=td)
            loaded = DirectoryStore.load(td)
            assert loaded.params == store.params
            assert loaded.zarr_store.path == store.zarr_store.path


class TestStoreRun:
    def test_memory_store_sample(self):
        store = MemoryStore.from_model(model, prior)
        indices = store.sample(100, prior)
        assert len(indices) == len(store)

    def test_memory_store_simulate(self):
        store = MemoryStore(2, simulator=sim_multi_out)
        indices = store.sample(100, prior)
        ind_sim = indices[:50]
        store.simulate(ind_sim)

        assert store.sims.x1[49].sum() != 0
        assert store.sims.x1[50].sum() == 0

    def test_directory_store_sample(self):
        with tempfile.TemporaryDirectory() as td:
            store = DirectoryStore(2, simulator=sim_multi_out, path=td)
            indices = store.sample(100, prior)
            assert len(indices) == len(store)

    def test_directory_store_simulate(self):
        with tempfile.TemporaryDirectory() as td:
            store = DirectoryStore(2, simulator=sim, path=td)
            ind_sim = store.sample(100, prior)
            store.simulate(ind_sim)
            assert store.sims.x[:].sum(axis=1).all()

    def test_directory_store_simulate_partial(self):
        with tempfile.TemporaryDirectory() as td:
            store = DirectoryStore(2, simulator=sim, path=td)
            ind_sim = store.sample(100, prior)
            ind_sim = ind_sim[:40]
            store.simulate(ind_sim)
            assert store.sims.x[39].sum() != 0
            assert store.sims.x[40].sum() == 0

    def test_store_lockfile(self):
        with tempfile.TemporaryDirectory() as td:
            store_dir = DirectoryStore(
                2, simulator=sim, path=td, sync_path=td + ".sync"
            )
            assert store_dir._lock is not None
            assert store_dir._lock.lockfile is None

            store_dir.lock()
            assert store_dir._lock.lockfile is not None

            store_dir.unlock()
            assert store_dir._lock.lockfile is None


if __name__ == "__main__":
    pass
