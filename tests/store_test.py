import tempfile
from pathlib import Path

import numpy as np
import zarr

from swyft import Prior
from swyft.store.simulator import DaskSimulator, SimulationStatus, Simulator
from swyft.store.store import DirectoryStore, MemoryStore


def model(params):
    p = np.linspace(-1, 1, 10)
    a, b = params
    mu = p * a + b
    return dict(x=mu)


def broken_model(params):
    raise ValueError("oops!")


def model_multi_out(params):
    p = np.linspace(-1, 1, 10)
    a, b = params
    mu = p * a + b
    mu2 = np.reshape(p * a - b, (2, 5))
    return dict(x1=mu, x2=mu2)


sim = Simulator(model, pnames=["a", "b"], sim_shapes=dict(x=(10,)))
sim_multi_out = Simulator(
    model_multi_out, pnames=["a", "b"], sim_shapes=dict(x1=(10,), x2=(2, 5))
)
prior = Prior(lambda u: u * np.array([1.0, 0.5]), zdim=2)


class TestStoreIO:
    def test_init_memory_store(self):
        store = MemoryStore(simulator=sim)
        assert store.v.shape[1] == 2
        assert isinstance(store._zarr_store, zarr.storage.MemoryStore)
        assert isinstance(store._simulator, Simulator)

    def test_init_memory_store_multi_outputs(self):
        store = MemoryStore(simulator=sim_multi_out)
        assert store.v.shape[1] == 2
        assert {k: v.shape[1:] for k, v in store.sims.items()} == {
            "x1": (10,),
            "x2": (2, 5),
        }

    def test_init_directory_store_multi_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            store = DirectoryStore(simulator=sim_multi_out, path=td)
            assert store.v.shape[1] == 2
            assert {k: v.shape[1:] for k, v in store.sims.items()} == {
                "x1": (10,),
                "x2": (2, 5),
            }
            td_path = Path(td)
            items = [
                p.relative_to(td).as_posix() for p in td_path.rglob("*/") if p.is_dir()
            ]
            assert len(items) > 0

    def test_memory_store_save(self):
        store = MemoryStore(simulator=sim_multi_out)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            store.save(td)
            items = [
                p.relative_to(td).as_posix() for p in td_path.rglob("*") if p.is_dir()
            ]
            assert len(items) > 0

    def test_memory_store_load(self):
        store = MemoryStore(simulator=sim_multi_out)
        with tempfile.TemporaryDirectory() as td:
            store.save(td)
            loaded = MemoryStore.load(td)
            loaded.set_simulator(sim_multi_out)
            assert np.allclose(loaded.v, store.v)
            assert loaded._zarr_store.root == store._zarr_store.root
            assert all([np.allclose(loaded.sims[k], v) for k, v in store.sims.items()])

    def test_directory_store_load(self):
        with tempfile.TemporaryDirectory() as td:
            store = DirectoryStore(simulator=sim_multi_out, path=td)
            loaded = MemoryStore.load(td)
            assert np.allclose(loaded.v, store.v)


class TestStoreRun:
    def test_store_add(self):
        store = MemoryStore(simulator=sim_multi_out)
        store.add(20, prior)
        assert store.sims.x1.shape[0] > 0

    def test_memory_store_simulate(self):
        store = MemoryStore(simulator=sim_multi_out)
        indices = store.sample(100, prior, add=True)
        ind_sim = indices[:50]
        store.simulate(ind_sim)

        assert store.sims.x1[49].sum() != 0
        assert store.sims.x1[50].sum() == 0

    def test_directory_store_sample(self):
        with tempfile.TemporaryDirectory() as td:
            store = DirectoryStore(simulator=sim_multi_out, path=td)
            indices = store.sample(100, prior, add=True)
            assert len(indices) == len(store)

    def test_directory_store_simulate(self):
        with tempfile.TemporaryDirectory() as td:
            store = DirectoryStore(simulator=sim, path=td)
            ind_sim = store.sample(100, prior, add=True)
            store.simulate(ind_sim)
            assert store.sims.x[:].sum(axis=1).all()

    def test_directory_store_simulate_partial(self):
        with tempfile.TemporaryDirectory() as td:
            store = DirectoryStore(simulator=sim, path=td)
            ind_sim = store.sample(100, prior, add=True)
            ind_sim = ind_sim[:40]
            store.simulate(ind_sim)
            assert store.sims.x[39].sum() != 0
            assert store.sims.x[40].sum() == 0

    def test_store_lockfile(self):
        with tempfile.TemporaryDirectory() as td:
            store_dir = DirectoryStore(simulator=sim, path=td, sync_path=td + ".sync")
            assert store_dir._lock is not None
            assert store_dir._lock.lockfile is None

            store_dir.lock()
            assert store_dir._lock.lockfile is not None

            store_dir.unlock()
            assert store_dir._lock.lockfile is None

    def test_interrupted_simulator_failed(self):
        sim_fail = Simulator(broken_model, pnames=["a", "b"], sim_shapes={"obs": (2,)})
        store = MemoryStore(simulator=sim_fail)
        store.add(10, prior)
        store.simulate()
        assert all(store.sim_status[:] == SimulationStatus.FAILED)

    def test_interrupted_dasksimulator_failed(self):
        sim_fail = DaskSimulator(
            broken_model, pnames=["a", "b"], sim_shapes={"obs": (2,)}
        )
        store = MemoryStore(simulator=sim_fail)
        # store = DirectoryStore(path="test.zarr", simulator=sim)
        store.add(10, prior)
        store.simulate()
        assert all(store.sim_status[:] == SimulationStatus.FAILED)
