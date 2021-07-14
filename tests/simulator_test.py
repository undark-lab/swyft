import tempfile
import time

import numpy as np
import zarr
from dask.distributed import LocalCluster, get_client

from swyft import Simulator
from swyft.store.simulator import SimulationStatus


def model(params):
    p = np.linspace(-1, 1, 10)
    a, b = params
    mu = p * a + b
    return dict(x=mu)


def _wait_for_all_tasks(timeout=20):
    client = get_client()
    start_time = time.time()
    while time.time() - start_time < timeout:
        if len(client.who_has()) == 0:
            break
        time.sleep(0.1)


def test_run_simulator_with_processes_and_numpy_array():
    """
    If the store is in memory (here a Numpy array) and the Dask workers do not
    share  memory with the client (e.g. we have a processes-based cluster),
    collect_in_memory must be set to True.
    """
    cluster = LocalCluster(n_workers=2, processes=True, threads_per_worker=1)
    simulator = Simulator(model, sim_shapes=dict(x=(10,)), cluster=cluster)

    pars = np.random.random((100, 2))
    sims = dict(x=np.zeros((100, 10)))
    sim_status = np.full(100, SimulationStatus.RUNNING, dtype=np.int)

    simulator.run(
        pars=pars,
        sims=sims,
        sim_status=sim_status,
        indices=np.arange(100, dtype=np.int),
        collect_in_memory=True,
        batch_size=20,
    )

    assert np.all(sim_status == SimulationStatus.FINISHED)
    assert not np.all(np.isclose(sims["x"].sum(axis=1), 0.0))
    simulator.client.close()
    cluster.close()


def test_run_simulator_with_threads_and_numpy_array():
    """
    If the store is in memory (here a Numpy array) and the Dask workers share
    memory with the client (i.e. we have a threads-based cluster),
    collect_in_memory can be set to False.
    """
    cluster = LocalCluster(n_workers=2, processes=False, threads_per_worker=1)
    simulator = Simulator(model, sim_shapes=dict(x=(10,)), cluster=cluster)

    pars = np.random.random((100, 2))
    sims = dict(x=np.zeros((100, 10)))
    sim_status = np.full(100, SimulationStatus.RUNNING, dtype=np.int)

    # the following is non-blocking (it immediately returns)
    simulator.run(
        pars=pars,
        sims=sims,
        sim_status=sim_status,
        indices=np.arange(100, dtype=np.int),
        collect_in_memory=False,
        batch_size=20,
    )

    # need to wait for tasks to be completed
    _wait_for_all_tasks()

    assert np.all(sim_status == SimulationStatus.FINISHED)
    assert not np.all(np.isclose(sims["x"].sum(axis=1), 0.0))
    simulator.client.close()
    cluster.close()


def test_run_simulator_with_processes_and_zarr_memory_store():
    """
    If the store is in memory (here a Zarr MemoryStore) and the Dask workers do
    not share memory with the client (i.e. we have a processes-based cluster),
    collect_in_memory must be set to True.
    """
    cluster = LocalCluster(n_workers=2, processes=True, threads_per_worker=1)
    simulator = Simulator(model, sim_shapes=dict(x=(10,)), cluster=cluster)

    pars = zarr.zeros((100, 2))
    pars[:, :] = np.random.random(pars.shape)
    x = zarr.zeros((100, 10))
    sims = dict(x=x.oindex)
    sim_status = zarr.full(100, SimulationStatus.RUNNING, dtype="int")

    simulator.run(
        pars=pars,
        sims=sims,
        sim_status=sim_status.oindex,
        indices=np.arange(100, dtype=np.int),
        collect_in_memory=True,
        batch_size=20,
    )

    assert np.all(sim_status[:] == SimulationStatus.FINISHED)
    assert not np.all(np.isclose(sims["x"][:, :].sum(axis=1), 0.0))
    simulator.client.close()
    cluster.close()


def test_run_simulator_with_threads_and_zarr_memory_store():
    """
    If the store is in memory (here a Zarr MemoryStore) and the Dask workers
    share memory with the client (i.e. we have a threads-based cluster),
    collect_in_memory can be set to False.
    """
    cluster = LocalCluster(n_workers=2, processes=False, threads_per_worker=1)
    simulator = Simulator(model, sim_shapes=dict(x=(10,)), cluster=cluster)

    pars = zarr.zeros((100, 2))
    pars[:, :] = np.random.random(pars.shape)
    x = zarr.zeros((100, 10))
    sims = dict(x=x.oindex)
    sim_status = zarr.full(100, SimulationStatus.RUNNING, dtype="int")

    # the following is non-blocking (it immediately returns)
    simulator.run(
        pars=pars,
        sims=sims,
        sim_status=sim_status.oindex,
        indices=np.arange(100, dtype=np.int),
        collect_in_memory=False,
        batch_size=20,
    )

    # need to wait for tasks to be completed
    _wait_for_all_tasks()

    assert np.all(sim_status[:] == SimulationStatus.FINISHED)
    assert not np.all(np.isclose(sims["x"][:, :].sum(axis=1), 0.0))
    simulator.client.close()
    cluster.close()


def test_run_simulator_with_threads_and_zarr_directory_store():
    """
    If the store is on disk (here a Zarr DirectoryStore), collect_in_memory can
    be set to False (but synchronization needs to be employed).
    """
    cluster = LocalCluster(n_workers=2, processes=False, threads_per_worker=1)
    simulator = Simulator(model, sim_shapes=dict(x=(10,)), cluster=cluster)

    with tempfile.TemporaryDirectory() as tmpdir:
        pars = zarr.open(f"{tmpdir}/pars.zarr", shape=(100, 2))
        pars[:, :] = np.random.random(pars.shape)
        x = zarr.open(
            f"{tmpdir}/x.zarr", shape=(100, 10), synchronizer=zarr.ThreadSynchronizer()
        )
        x[:, :] = 0.0
        sims = dict(x=x.oindex)
        sim_status = zarr.open(
            f"{tmpdir}/sim_status.zarr",
            shape=(100,),
            synchronizer=zarr.ThreadSynchronizer(),
        )
        sim_status[:] = np.full(100, SimulationStatus.RUNNING, dtype="int")

        # the following is non-blocking (it immediately returns)
        simulator.run(
            pars=pars,
            sims=sims,
            sim_status=sim_status.oindex,
            indices=np.arange(100, dtype=np.int),
            collect_in_memory=False,
            batch_size=20,
        )

        # need to wait for tasks to be completed
        _wait_for_all_tasks()

        assert np.all(sim_status[:] == SimulationStatus.FINISHED)
        assert not np.all(np.isclose(sims["x"][:, :].sum(axis=1), 0.0))
    simulator.client.close()
    cluster.close()


def test_run_simulator_with_processes_and_zarr_directory_store():
    """
    If the store is on disk (here a Zarr DirectoryStore), collect_in_memory can
    be set to False (but synchronization needs to be employed).
    """
    cluster = LocalCluster(n_workers=2, processes=True, threads_per_worker=1)
    simulator = Simulator(model, sim_shapes=dict(x=(10,)), cluster=cluster)

    with tempfile.TemporaryDirectory() as tmpdir:
        pars = zarr.open(f"{tmpdir}/pars.zarr", shape=(100, 2))
        pars[:, :] = np.random.random(pars.shape)
        synchronizer = zarr.ProcessSynchronizer(path=f"{tmpdir}/x.sync")
        x = zarr.open(f"{tmpdir}/x.zarr", shape=(100, 10), synchronizer=synchronizer)
        x[:, :] = 0.0
        sims = dict(x=x.oindex)
        synchronizer = zarr.ProcessSynchronizer(path=f"{tmpdir}/sim_status.sync")
        sim_status = zarr.open(
            f"{tmpdir}/sim_status.zarr",
            shape=(100,),
            synchronizer=synchronizer,
            dtype="int",
        )
        sim_status[:] = np.full(100, SimulationStatus.RUNNING, dtype="int")

        # the following is non-blocking (it immediately returns)
        simulator.run(
            pars=pars,
            sims=sims,
            sim_status=sim_status.oindex,
            indices=np.arange(100, dtype=np.int),
            collect_in_memory=False,
            batch_size=20,
        )

        # need to wait for tasks to be completed
        _wait_for_all_tasks()

        assert np.all(sim_status[:] == SimulationStatus.FINISHED)
        assert not np.all(np.isclose(sims["x"][:, :].sum(axis=1), 0.0))
    simulator.client.close()
    cluster.close()
