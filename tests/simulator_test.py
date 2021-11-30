import decimal
import tempfile
import time
import unittest

import numpy as np
import zarr
from dask.distributed import LocalCluster, get_client

from swyft import DaskSimulator, Simulator
from swyft.prior import get_uniform_prior
from swyft.store.simulator import SimulationStatus


def model(params):
    p = np.linspace(-1, 1, 10)
    a, b = params
    mu = p * a + b
    return dict(x=mu)


def model_fail_if_negative(param):
    if param < 1:
        raise ValueError
    else:
        mu = param
    return dict(x=mu)


def model_inf_if_negative(param):
    if param < 1:
        mu = np.inf
    else:
        mu = param
    return dict(x=mu)


def _wait_for_all_tasks(timeout=20):
    client = get_client()
    start_time = time.time()
    while time.time() - start_time < timeout:
        if len(client.who_has()) == 0:
            break
        time.sleep(0.1)


class TestSimulator(unittest.TestCase):
    def test_run_simulator_all_pass(self):
        simulator = Simulator(model, sim_shapes=dict(x=(10,)), parameter_names=2)
        pars = np.random.random((100, 2))
        sims = dict(x=np.zeros((100, 10)))
        sim_status = np.full(100, SimulationStatus.RUNNING, dtype=int)

        simulator._run(
            v=pars,
            sims=sims,
            sim_status=sim_status,
            indices=np.arange(100, dtype=int),
        )

        assert np.all(sim_status == SimulationStatus.FINISHED)
        assert not np.all(np.isclose(sims["x"].sum(axis=1), 0.0))

    def test_run_simulator_fail_on_wrong_sim_shape(self):
        simulator = Simulator(model, sim_shapes=dict(x=(11,)), parameter_names=2)
        pars = np.random.random((100, 2))
        sims = dict(x=np.zeros((100, 11)))
        sim_status = np.full(100, SimulationStatus.RUNNING, dtype=int)

        self.assertRaises(
            ValueError,
            simulator._run,
            pars,
            sims,
            sim_status,
            np.arange(100, dtype=int),
        )

    def test_run_simulator_fail_on_wrong_num_params(self):
        simulator = Simulator(model, sim_shapes=dict(x=(10,)), parameter_names=1)
        pars = np.random.random((100, 1))
        sims = dict(x=np.zeros((100, 10)))
        sim_status = np.full(100, SimulationStatus.RUNNING, dtype=int)

        simulator._run(
            v=pars,
            sims=sims,
            sim_status=sim_status,
            indices=np.arange(100, dtype=int),
        )

        assert np.all(sim_status == SimulationStatus.FAILED)

    def test_run_simulator_partial_fail(self):
        simulator = Simulator(
            model_fail_if_negative, sim_shapes=dict(x=(1,)), parameter_names=1
        )
        pars = np.ones((100,))
        pars[0:49] = -1
        sims = dict(x=np.zeros((100, 1)))
        sim_status = np.full(100, SimulationStatus.RUNNING, dtype=int)

        simulator._run(
            v=pars,
            sims=sims,
            sim_status=sim_status,
            indices=np.arange(100, dtype=int),
        )

        assert np.all(sim_status[:49] == SimulationStatus.FAILED)
        assert np.all(sim_status[50:] == SimulationStatus.FINISHED)

    def test_run_simulator_partial_fail_on_inf(self):
        simulator = Simulator(
            model_inf_if_negative,
            sim_shapes=dict(x=(1,)),
            parameter_names=1,
            fail_on_non_finite=True,
        )
        pars = np.ones((100,))
        pars[0:49] = -1
        sims = dict(x=np.zeros((100, 1)))
        sim_status = np.full(100, SimulationStatus.RUNNING, dtype=int)

        simulator._run(
            v=pars,
            sims=sims,
            sim_status=sim_status,
            indices=np.arange(100, dtype=int),
        )

        assert np.all(sim_status[:49] == SimulationStatus.FAILED)
        assert np.all(sim_status[50:] == SimulationStatus.FINISHED)

    def test_run_simulator_pass_on_inf(self):
        simulator = Simulator(
            model_inf_if_negative,
            sim_shapes=dict(x=(1,)),
            parameter_names=1,
            fail_on_non_finite=False,
        )
        pars = np.ones((100,))
        pars[0:49] = -1
        sims = dict(x=np.zeros((100, 1)))
        sim_status = np.full(100, SimulationStatus.RUNNING, dtype=int)

        simulator._run(
            v=pars,
            sims=sims,
            sim_status=sim_status,
            indices=np.arange(100, dtype=int),
        )

        assert np.all(sim_status[50:] == SimulationStatus.FINISHED)

    def test_run_simulator_with_zarr_memory_store(self):
        """Test the simulator with a store based on Zarr MemoryStore."""
        simulator = Simulator(model, sim_shapes=dict(x=(10,)), parameter_names=2)

        pars = zarr.zeros((100, 2))
        pars[:, :] = np.random.random(pars.shape)
        x = zarr.zeros((100, 10))
        sims = dict(x=x.oindex)
        sim_status = zarr.full(100, SimulationStatus.RUNNING, dtype="int")

        simulator._run(
            v=pars,
            sims=sims,
            sim_status=sim_status.oindex,
            indices=np.arange(100, dtype=int),
        )

        assert np.all(sim_status[:] == SimulationStatus.FINISHED)
        assert not np.all(np.isclose(sims["x"][:, :].sum(axis=1), 0.0))

    def test_run_simulator_with_zarr_directory_store(self):
        """Test simulator with a store based on Zarr DirectoryStore."""
        simulator = Simulator(model, sim_shapes=dict(x=(10,)), parameter_names=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            pars = zarr.open(f"{tmpdir}/pars.zarr", shape=(100, 2))
            pars[:, :] = np.random.random(pars.shape)
            x = zarr.open(f"{tmpdir}/x.zarr", shape=(100, 10))
            x[:, :] = 0.0
            sims = dict(x=x.oindex)
            sim_status = zarr.open(f"{tmpdir}/sim_status.zarr", shape=(100,))
            sim_status[:] = np.full(100, SimulationStatus.RUNNING, dtype="int")

            simulator._run(
                v=pars,
                sims=sims,
                sim_status=sim_status.oindex,
                indices=np.arange(100, dtype=int),
            )

            assert np.all(sim_status[:] == SimulationStatus.FINISHED)
            assert not np.all(np.isclose(sims["x"][:, :].sum(axis=1), 0.0))

    def test_setup_simulator_from_model_function(self):
        dimension = 2
        prior = get_uniform_prior(-1 * np.ones(dimension), np.ones(dimension))
        simulator = Simulator.from_model(model, prior)
        assert len(simulator.parameter_names) == 2
        assert simulator.sim_shapes == {"x": (10,)}

    def test_run_a_simulator_that_is_setup_from_command_line(self):
        def set_input(v):
            ctx = decimal.Context()
            ctx.prec = 16
            return "{} + {}\n".format(
                ctx.create_decimal(repr(v[0])), ctx.create_decimal(repr(v[1]))
            )

        def get_output(stdout, _):
            return {"sum": float(stdout)}

        simulator = Simulator.from_command(
            command="bc -l",
            set_input_method=set_input,
            get_output_method=get_output,
            parameter_names=2,
            sim_shapes={"sum": ()},
        )

        pars = np.random.random((100, 2))
        sims = dict(sum=np.zeros(100))
        sim_status = np.full(100, SimulationStatus.RUNNING, dtype=int)

        simulator._run(
            v=pars,
            sims=sims,
            sim_status=sim_status,
            indices=np.arange(100, dtype=int),
        )

        assert np.all(sim_status == SimulationStatus.FINISHED)
        assert not np.any(np.isclose(sims["sum"], 0.0))


class TestDaskSimulator(unittest.TestCase):
    def test_connect_to_dask_simulator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with LocalCluster(
                n_workers=2,
                processes=True,
                threads_per_worker=1,
                local_directory=tmpdir,
            ) as cluster:
                simulator = DaskSimulator(
                    model_fail_if_negative, sim_shapes=dict(x=(10,)), parameter_names=2
                )
                simulator.set_dask_cluster(cluster)
                assert cluster.name == simulator.client.cluster.name

    def test_run_simulator_with_processes_and_numpy_array(self):
        """
        If the store is in memory (here a Numpy array) and the Dask workers do not
        share  memory with the client (e.g. we have a processes-based cluster),
        collect_in_memory must be set to True.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with LocalCluster(
                n_workers=2,
                processes=True,
                threads_per_worker=1,
                local_directory=tmpdir,
            ) as cluster:
                simulator = DaskSimulator(
                    model, sim_shapes=dict(x=(10,)), parameter_names=2
                )
                simulator.set_dask_cluster(cluster)

                pars = np.random.random((100, 2))
                sims = dict(x=np.zeros((100, 10)))
                sim_status = np.full(100, SimulationStatus.RUNNING, dtype=int)

                simulator._run(
                    v=pars,
                    sims=sims,
                    sim_status=sim_status,
                    indices=np.arange(100, dtype=int),
                    collect_in_memory=True,
                    batch_size=20,
                )

                assert np.all(sim_status == SimulationStatus.FINISHED)
                assert not np.all(np.isclose(sims["x"].sum(axis=1), 0.0))

    def test_run_simulator_with_processes_and_zarr_memory_store(self):
        """
        If the store is in memory (here a Zarr MemoryStore) and the Dask workers do
        not share memory with the client (i.e. we have a processes-based cluster),
        collect_in_memory must be set to True.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with LocalCluster(
                n_workers=2,
                processes=True,
                threads_per_worker=1,
                local_directory=tmpdir,
            ) as cluster:
                simulator = DaskSimulator(
                    model, sim_shapes=dict(x=(10,)), parameter_names=2
                )
                simulator.set_dask_cluster(cluster)

                pars = zarr.zeros((100, 2))
                pars[:, :] = np.random.random(pars.shape)
                x = zarr.zeros((100, 10))
                sims = dict(x=x.oindex)
                sim_status = zarr.full(100, SimulationStatus.RUNNING, dtype="int")

                simulator._run(
                    v=pars,
                    sims=sims,
                    sim_status=sim_status.oindex,
                    indices=np.arange(100, dtype=int),
                    collect_in_memory=True,
                    batch_size=20,
                )

                assert np.all(sim_status[:] == SimulationStatus.FINISHED)
                assert not np.all(np.isclose(sims["x"][:, :].sum(axis=1), 0.0))

    def test_run_simulator_with_threads_and_zarr_directory_store(self):
        """
        If the store is on disk (here a Zarr DirectoryStore), collect_in_memory can
        be set to False (but synchronization needs to be employed).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with LocalCluster(
                n_workers=2,
                processes=False,
                threads_per_worker=1,
                local_directory=tmpdir,
            ) as cluster:
                simulator = DaskSimulator(
                    model, sim_shapes=dict(x=(10,)), parameter_names=2
                )
                simulator.set_dask_cluster(cluster)
                pars = zarr.open(f"{tmpdir}/pars.zarr", shape=(100, 2))
                pars[:, :] = np.random.random(pars.shape)
                x = zarr.open(
                    f"{tmpdir}/x.zarr",
                    shape=(100, 10),
                    synchronizer=zarr.ThreadSynchronizer(),
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
                simulator._run(
                    v=pars,
                    sims=sims,
                    sim_status=sim_status.oindex,
                    indices=np.arange(100, dtype=int),
                    collect_in_memory=False,
                    batch_size=20,
                )

                # need to wait for tasks to be completed
                _wait_for_all_tasks()

                assert np.all(sim_status[:] == SimulationStatus.FINISHED)
                assert not np.all(np.isclose(sims["x"][:, :].sum(axis=1), 0.0))

    def test_run_simulator_with_processes_and_zarr_directory_store(self):
        """
        If the store is on disk (here a Zarr DirectoryStore), collect_in_memory can
        be set to False (but synchronization needs to be employed).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with LocalCluster(
                n_workers=2,
                processes=True,
                threads_per_worker=1,
                local_directory=tmpdir,
            ) as cluster:
                simulator = DaskSimulator(
                    model, sim_shapes=dict(x=(10,)), parameter_names=2
                )
                simulator.set_dask_cluster(cluster)
                pars = zarr.open(f"{tmpdir}/pars.zarr", shape=(100, 2))
                pars[:, :] = np.random.random(pars.shape)
                synchronizer = zarr.ProcessSynchronizer(path=f"{tmpdir}/x.sync")
                x = zarr.open(
                    f"{tmpdir}/x.zarr", shape=(100, 10), synchronizer=synchronizer
                )
                x[:, :] = 0.0
                sims = dict(x=x.oindex)
                synchronizer = zarr.ProcessSynchronizer(
                    path=f"{tmpdir}/sim_status.sync"
                )
                sim_status = zarr.open(
                    f"{tmpdir}/sim_status.zarr",
                    shape=(100,),
                    synchronizer=synchronizer,
                    dtype="int",
                )
                sim_status[:] = np.full(100, SimulationStatus.RUNNING, dtype="int")

                # the following is non-blocking (it immediately returns)
                simulator._run(
                    v=pars,
                    sims=sims,
                    sim_status=sim_status.oindex,
                    indices=np.arange(100, dtype=int),
                    collect_in_memory=False,
                    batch_size=20,
                )

                # need to wait for tasks to be completed
                _wait_for_all_tasks()

                assert np.all(sim_status[:] == SimulationStatus.FINISHED)
                assert not np.all(np.isclose(sims["x"][:, :].sum(axis=1), 0.0))

    def test_run_a_simulator_that_is_setup_from_command_line(self):
        """Run a simulator based on a command line model

        Notes:
            Need to have a cluster which uses only processes since each
            instance of the model is run in a separate subdirectory.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with LocalCluster(
                n_workers=2,
                processes=True,
                threads_per_worker=1,
                local_directory=tmpdir,
            ) as cluster:

                def set_input(v):
                    ctx = decimal.Context()
                    ctx.prec = 16
                    return "{} + {}\n".format(
                        ctx.create_decimal(repr(v[0])), ctx.create_decimal(repr(v[1]))
                    )

                def get_output(stdout, _):
                    return {"sum": float(stdout)}

                simulator = DaskSimulator.from_command(
                    command="bc -l",
                    set_input_method=set_input,
                    get_output_method=get_output,
                    parameter_names=2,
                    sim_shapes={"sum": ()},
                )
                simulator.set_dask_cluster(cluster)

                pars = np.random.randint(1, 100, size=(100, 2))
                sims = dict(sum=np.zeros(100))
                sim_status = np.full(100, SimulationStatus.RUNNING, dtype=int)

                simulator._run(
                    v=pars,
                    sims=sims,
                    sim_status=sim_status,
                    indices=np.arange(100, dtype=int),
                )

                assert np.all(sim_status == SimulationStatus.FINISHED)
                assert not np.any(np.isclose(sims["sum"], 0.0))
