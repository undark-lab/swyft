import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pytest
import zarr
from dask.distributed import LocalCluster

from swyft import Dataset, DirectoryStore, Prior, Simulator
from swyft.store.simulator import SimulationStatus

PARAMS = ["z1", "z2"]
PRIOR = Prior.from_uv(lambda u: u * np.array([1 for _ in PARAMS]), len(PARAMS))
OUTPUT_SHAPE = (20, 20)
SIM_SHAPES = {"x": OUTPUT_SHAPE}
N_SIMULATIONS = 1000
BATCH_SIZE = 100

MAX_WORKERS = 4  # number of simultaneous processes acting on the store


def model(_):
    """Model with dummy parameters. Return random numbers in (0; 1]."""
    return dict(x=-np.random.random(OUTPUT_SHAPE) + 1)


@pytest.fixture(scope="function")
def store():
    simulator = Simulator(model, sim_shapes=SIM_SHAPES)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_store"
        yield DirectoryStore(path=path, params=PARAMS, simulator=simulator)


@pytest.fixture(scope="module")
def cluster():
    return LocalCluster(n_workers=2, threads_per_worker=1)


def simulate(cluster, path="./cache", wait_for_results=True):
    """
    Open store, sample simulation parameters and run the corresponding
    simulations.
    """
    simulator = Simulator(model=model, sim_shapes=SIM_SHAPES, cluster=cluster)
    store = DirectoryStore(path=path, params=PARAMS, simulator=simulator)
    dataset = Dataset(N_SIMULATIONS, PRIOR, store=store)
    dataset.simulate(wait_for_results=wait_for_results, batch_size=BATCH_SIZE)
    return dataset.indices


def read_from_store(path):
    """Extract data from the Zarr Directory store"""
    z = zarr.open(f"{path}/samples/pars")
    x = zarr.open_group(f"{path}/samples/sims")
    s = zarr.open_array(f"{path}/samples/simulation_status")
    return z[:], {key: val[:] for key, val in x.items()}, s[:]


def test_concurrent_runs_waiting_for_results(cluster, store):
    """
    Run several processes that access the same store to sample parameters and
    to submit the corresponding simulations. The outcome of the simulations
    is waited for within the processes, so when they return all outcome should
    be written to the store.
    """
    path = store._zarr_store.path
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(MAX_WORKERS):
            # each process grows and sample the same cache
            future = executor.submit(
                simulate, cluster=cluster.scheduler_address, path=path
            )
            futures.append(future)

        for future in as_completed(futures):
            # processes are waiting for results, so all simulations should be finished
            status = store.get_simulation_status(future.result())
            assert np.all(status == SimulationStatus.FINISHED)

    z, x, s = read_from_store(path)

    # check shape of the parameter array
    n_simulations, n_params = z.shape
    # the real number of samples can differ slightly from the required value
    assert n_simulations > 0.80 * N_SIMULATIONS and n_simulations < 1.20 * N_SIMULATIONS
    assert n_params == len(PARAMS)

    # check shape and values of the simulation array
    assert x.keys() == SIM_SHAPES.keys()
    for key, val in SIM_SHAPES.items():
        assert x[key].shape == (n_simulations, *val)
        assert np.all(x[key][:] > 0.0)  # all simulation output has been updated

    # check shape and values of the status array
    assert s.shape == (n_simulations,)
    assert np.all(s == SimulationStatus.FINISHED)  # all simulations are done


def test_concurrent_run_without_waiting_for_results(cluster, store):
    """
    Run several processes that access the same store to sample parameters and
    to submit the corresponding simulations. The processes do not wait for the
    simulations to be done, so when they return some simulations should still
    be running.
    """

    path = store._zarr_store.path
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(MAX_WORKERS):
            # each process grows and sample the same cache
            future = executor.submit(
                simulate,
                cluster=cluster.scheduler_address,
                path=path,
                wait_for_results=False,
            )
            futures.append(future)

        for future in as_completed(futures):
            # processes are not waiting for results, so some simulations should still be running
            status = store.get_simulation_status(future.result())
            assert np.any(status == SimulationStatus.RUNNING)

    z, x, s = read_from_store(path)

    # check shape of the parameter array
    n_simulations, n_params = z.shape
    # the real number of samples can differ slightly from the required value
    assert n_simulations > 0.80 * N_SIMULATIONS and n_simulations < 1.20 * N_SIMULATIONS
    assert n_params == len(PARAMS)

    # check shape of the simulation array
    assert x.keys() == SIM_SHAPES.keys()
    for key, val in SIM_SHAPES.items():
        assert x[key].shape == (n_simulations, *val)

    # check shape of the status array
    assert s.shape == (n_simulations,)

    # now explicitly wait for simulations
    store.wait_for_simulations(indices=np.arange(n_simulations))

    z, x, s = read_from_store(path)
    for key, val in SIM_SHAPES.items():
        assert np.all(x[key] > 0.0)  # all simulation output has been updated

    assert np.all(s == SimulationStatus.FINISHED)  # all simulations are done
