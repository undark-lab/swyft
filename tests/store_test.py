import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr
from dask.distributed import LocalCluster

from swyft.prior import get_uniform_prior
from swyft.store.simulator import DaskSimulator, SimulationStatus, Simulator
from swyft.store.store import Store


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


sim = Simulator(model, parameter_names=["a", "b"], sim_shapes=dict(x=(10,)))
sim_multi_out = Simulator(
    model_multi_out, parameter_names=["a", "b"], sim_shapes=dict(x1=(10,), x2=(2, 5))
)
prior = get_uniform_prior(np.zeros(2), np.array([1.0, 0.5]))


class TestStoreIO:
    def test_init_memory_store(self):
        store = Store.memory_store(simulator=sim)
        assert store.v.shape[1] == 2
        assert isinstance(store._zarr_store, zarr.storage.MemoryStore)
        assert isinstance(store._simulator, Simulator)

    def test_init_memory_store_multi_outputs(self):
        store = Store.memory_store(simulator=sim_multi_out)
        assert store.v.shape[1] == 2
        assert {k: v.shape[1:] for k, v in store.sims.items()} == {
            "x1": (10,),
            "x2": (2, 5),
        }

    def test_init_directory_store_multi_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td) / "store.zarr"
            store = Store.directory_store(simulator=sim_multi_out, path=td_path)
            assert store.v.shape[1] == 2
            assert {k: v.shape[1:] for k, v in store.sims.items()} == {
                "x1": (10,),
                "x2": (2, 5),
            }
            groups = set(p.name for p in td_path.glob("*") if p.is_dir())
            assert groups == {"samples", "metadata"}
            sims = set(
                p.name for p in (td_path / "samples/sims/").glob("*") if p.is_dir()
            )
            assert sims == {"x1", "x2"}

    def test_init_directory_store_with_existing_path(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileExistsError):
                Store.directory_store(simulator=sim_multi_out, path=td)

    def test_init_directory_store_overwriting_existing_path(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store.directory_store(
                simulator=sim_multi_out, path=td, overwrite=True
            )
            assert store.v.shape[1] == 2
            assert {k: v.shape[1:] for k, v in store.sims.items()} == {
                "x1": (10,),
                "x2": (2, 5),
            }
            td_path = Path(td)
            groups = set(p.name for p in td_path.glob("*") if p.is_dir())
            assert groups == {"samples", "metadata"}
            sims = set(
                p.name for p in (td_path / "samples/sims/").glob("*") if p.is_dir()
            )
            assert sims == {"x1", "x2"}

    def test_memory_store_save(self):
        store = Store.memory_store(simulator=sim_multi_out)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td) / "store.zarr"
            store.save(td_path)
            groups = set(p.name for p in td_path.glob("*") if p.is_dir())
            assert groups == {"samples", "metadata"}
            sims = set(
                p.name for p in (td_path / "samples/sims/").glob("*") if p.is_dir()
            )
            assert sims == {"x1", "x2"}

    def test_directory_store_save_to_different_path(self):
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            td_from = Path(td1) / "store.zarr"
            store = Store.directory_store(simulator=sim_multi_out, path=td_from)
            td_to = Path(td2) / "store.zarr"
            store.save(td_to)
            groups = set(p.name for p in td_to.glob("*") if p.is_dir())
            assert groups == {"samples", "metadata"}
            sims = set(
                p.name for p in (td_to / "samples/sims/").glob("*") if p.is_dir()
            )
            assert sims == {"x1", "x2"}

    def test_directory_store_save_to_same_path_does_not_remove_store(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td) / "store.zarr"
            store = Store.directory_store(simulator=sim_multi_out, path=td_path)
            store.save(td_path)
            groups = set(p.name for p in td_path.glob("*") if p.is_dir())
            assert groups == {"samples", "metadata"}
            sims = set(
                p.name for p in (td_path / "samples/sims/").glob("*") if p.is_dir()
            )
            assert sims == {"x1", "x2"}

    def test_directory_store_load_existing_store(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td) / "store.zarr"
            store = Store.directory_store(simulator=sim_multi_out, path=td_path)
            loaded = Store.load(td_path)
            # loaded.set_simulator(sim_multi_out)
            assert np.allclose(loaded.v, store.v)
            assert loaded._zarr_store.path == store._zarr_store.path
            assert loaded.sims.keys() == store.sims.keys()

    def test_directory_store_load_store_from_wrong_paths(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(KeyError):
                # path exists, keys not found
                Store.load(td)
            with pytest.raises(FileNotFoundError):
                # non-existent path
                td_path = Path(td) / "store.zarr"
                Store.load(td_path)

    def test_directory_store_to_memory(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td) / "store.zarr"
            store = Store.directory_store(simulator=sim_multi_out, path=td_path)
            loaded = store.to_memory()
            assert isinstance(loaded._zarr_store, zarr.MemoryStore)
            assert np.allclose(loaded.v.shape, store.v.shape)


class TestStoreRun:
    def test_store_add(self):
        store = Store.memory_store(simulator=sim_multi_out)
        store.add(20, prior)
        assert store.sims.x1.shape[0] > 0

    def test_memory_store_simulate(self):
        store = Store.memory_store(simulator=sim_multi_out)
        indices = store.sample(100, prior, add=True)
        ind_sim = indices[:50]
        store.simulate(ind_sim)

        assert store.sims.x1[49].sum() != 0
        assert store.sims.x1[50].sum() == 0

    def test_directory_store_sample(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store.directory_store(
                simulator=sim_multi_out, path=td, overwrite=True
            )
            indices = store.sample(100, prior, add=True)
            assert len(indices) == len(store)

    def test_directory_store_simulate(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store.directory_store(simulator=sim, path=td, overwrite=True)
            ind_sim = store.sample(100, prior, add=True)
            store.simulate(ind_sim)
            assert store.sims.x[:].sum(axis=1).all()

    def test_directory_store_simulate_partial(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store.directory_store(simulator=sim, path=td, overwrite=True)
            ind_sim = store.sample(100, prior, add=True)
            ind_sim = ind_sim[:40]
            store.simulate(ind_sim)
            assert store.sims.x[39].sum() != 0
            assert store.sims.x[40].sum() == 0

    def test_store_lockfile(self):
        with tempfile.TemporaryDirectory() as td:
            store_dir = Store.directory_store(
                simulator=sim, path=td, sync_path=td + ".sync", overwrite=True
            )
            assert store_dir._lock is not None
            assert store_dir._lock.lockfile is None

            store_dir.lock()
            assert store_dir._lock.lockfile is not None

            store_dir.unlock()
            assert store_dir._lock.lockfile is None

    def test_interrupted_simulator_failed(self):
        sim_fail = Simulator(
            broken_model, parameter_names=["a", "b"], sim_shapes={"obs": (2,)}
        )
        store = Store.memory_store(simulator=sim_fail)
        store.add(10, prior)
        store.simulate()
        assert all(store.sim_status[:] == SimulationStatus.FAILED)

    def test_interrupted_dasksimulator_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with LocalCluster(local_directory=tmpdir) as cluster:
                sim_fail = DaskSimulator(
                    broken_model, parameter_names=["a", "b"], sim_shapes={"obs": (2,)}
                )
                sim_fail.set_dask_cluster(cluster)

                store = Store.memory_store(simulator=sim_fail)
                store.add(10, prior)
                store.simulate()
                assert all(store.sim_status[:] == SimulationStatus.FAILED)
