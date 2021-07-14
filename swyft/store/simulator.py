import enum
import os
import shlex
import subprocess
import tempfile
from operator import getitem
from typing import Callable, Hashable, List, Mapping, Optional, Union

import dask.array as da
import numpy as np
from dask.distributed import Client, fire_and_forget, wait

from swyft.bounds import Prior
from swyft.types import (
    Array,
    ForwardModelType,
    PathType,
    PNamesType,
    Shape,
    SimShapeType,
)
from swyft.utils import all_finite


class SimulationStatus(enum.IntEnum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3


class Simulator:
    """Wrapper class for simulator.

    Args:
        model (callable): Model function
        pnames (int or list): List of parameter names, or number of
            parameters (interpreted as 'z0', 'z1', ...)
        sim_shapes (dict): Dict describing model function output shapes.

    Examples::

        >>> def model(v):
        >>>     mu = sum(v)  # mu = x + y + z
        >>>     nu = np.array([v[1], 2*v[2]])  # nu = [y, 2*z]
        >>>     return dict(mu=mu, nu=nu)
        >>> simulator = swyft.Simulator(model, ["x", "y", "z"], sim_shapes=dict(mu=(1,), nu=(2,))
    """

    def __init__(
        self,
        model: ForwardModelType,
        pnames: Union[PNamesType, int],
        sim_shapes: SimShapeType,
    ) -> None:
        self.model = model
        if isinstance(pnames, int):
            pnames = ["z%i" % i for i in range(pnames)]
        self.pnames = pnames
        self.sim_shapes = sim_shapes

    def _run(self, v, sims, sim_status, indices, **kwargs) -> None:
        """Run the simulator on the input parameters.

        Args:
            v: array with all the input parameters. Should have shape
                (num. samples, num. parameters)
            sims: dictionary of arrays where to store the simulation output.
                All arrays should have the number of samples as the size of the
                first dimension
            sim_status: array where to store the simulation status (size should
                be equal to the number of samples)
            indices: indices of the samples that need to be run by the
                simulator
        """
        for i in indices:
            sim = self.model(v[i])
            for k in sims.keys():
                sims[k][i] = sim[k]
                sim_status[i] = SimulationStatus.FINISHED


class DaskSimulator:
    """Setup and run the simulator engine, powered by dask."""

    def __init__(
        self,
        model: Callable,
        sim_shapes: SimShapeType,
        fail_on_non_finite: bool = True,
        cluster=None,
    ) -> None:
        """Initiate Simulator using a python function.

        Args:
            model: simulator model function
            sim_shapes: map of simulator's output names to shapes
            fail_on_non_finite: whether return an invalid code if simulation
                returns NaN or infinite, default True
            cluster: cluster address or Cluster object from dask.distributed
                (default is LocalCluster)
        """
        self.model = model
        self.sim_shapes = sim_shapes
        self.fail_on_non_finite = fail_on_non_finite
        self.client = None
        self.cluster = cluster

    def _run(
        self,
        v,
        sims,
        sim_status,
        indices,
        collect_in_memory: bool = True,
        batch_size: Optional[int] = None,
    ) -> None:  # TODO typing
        """Run the simulator on the input parameters.

        Args:
            v: array with all the input parameters. Should have shape
                (num. samples, num. parameters)
            sims: dictionary of arrays where to store the simulation output.
                All arrays should have the number of samples as the size of the
                first dimension
            sim_status: array where to store the simulation status (size should
                be equal to the number of samples)
            indices: indices of the samples that need to be run by the
                simulator
            collect_in_memory: if True, collect the simulation output in
                memory; if False, instruct Dask workers to save the output to
                the corresponding arrays. The latter option is asynchronous,
                thus this method immediately returns.
            batch_size: simulations will be submitted in batches of the
                specified size
        """
        self.set_dask_cluster(self.cluster)

        # open parameter array as Dask array
        chunks = getattr(v, "chunks", "auto")
        z = da.from_array(v, chunks=chunks)
        idx = da.from_array(indices, chunks=(batch_size or -1,))
        z = z[idx]

        z = z.persist()  # load the parameters in the distributed memory

        # block-wise run the model function on the parameter array
        out = da.map_blocks(
            _run_model_chunk,
            z,
            model=self.model,
            sim_shapes=self.sim_shapes,
            fail_on_non_finite=self.fail_on_non_finite,
            drop_axis=1,
            dtype=np.object,
        )

        # FIXME: Deprecated?
        #        print("Simulator: Running...")
        #        bag = db.from_sequence(z, npartitions=npartitions)
        #        bag = bag.map(_run_one_sample, self.model, self.fail_on_non_finite)
        #        result = bag.compute(scheduler=self.client or "processes")
        #        print("Simulator: ...done.")
        #        return result

        # split result dictionary and simulation status array
        results = out.map_blocks(getitem, 0, dtype=np.object)
        status = out.map_blocks(getitem, 1, meta=np.array(()), dtype=np.int)

        # unpack array of dictionaries to dictionary of arrays
        result_dict = {}
        for obs, shape in self.sim_shapes.items():
            result_dict[obs] = results.map_blocks(
                getitem,
                obs,
                new_axis=[i + 1 for i in range(len(shape))],
                chunks=(z.chunks[0], *shape),
                meta=np.array(()),
                dtype=np.float,
            )

        sources = [result_dict[k] for k in self.sim_shapes.keys()]
        targets = [sims[k] for k in self.sim_shapes.keys()]

        if collect_in_memory:
            # submit computation and collect results
            *sources, status = self.client.compute([*sources, status], sync=True)

            # update simulation results
            for source, target in zip(sources, targets):
                target[indices.tolist()] = source

            # finally, update the simulation status
            sim_status[indices.tolist()] = status

        else:
            sources = da.store(
                sources=sources,
                targets=targets,
                regions=(indices.tolist(),),
                lock=False,
                compute=False,
                return_stored=True,
            )

            # submit computation
            *sources, status = self.client.persist([*sources, status])

            # the following dummy array is generated after results are stored.
            zeros_when_done = [
                source.map_blocks(
                    lambda x: np.zeros(x.shape[0], dtype=np.int),
                    chunks=(source.chunks[0],),
                    drop_axis=[i for i in range(1, source.ndim)],
                    meta=np.array((), dtype=np.int),
                    dtype=np.int,
                )
                for source in sources
            ]
            status = sum([*zeros_when_done, status])
            status = status.store(
                target=sim_status,
                regions=(indices.tolist(),),
                lock=False,
                compute=False,
                return_stored=True,
            )
            # when the simulation results are stored, we can update the status
            status = self.client.persist(status)
            fire_and_forget(status)

    @classmethod
    def from_command(
        cls,
        command: str,
        set_input_method: Callable,
        get_output_method: Callable,
        tmpdir: PathType = None,
        **kwargs,
    ):  # TODO typing
        """Convenience function to setup a command-line simulator

        Args:
            command: command line simulator
            set_input_method: method to prepare simulator input
            get_output_method: method to retrieve results from the simulator output
            tmpdir: temporary directory where to run the simulator instances
                (one in each subdir). tmpdir must exist.
        """
        command_args = shlex.split(command)

        def model(z):
            """
            Closure to run an instance of the simulator

            Args:
                z (array-like): input parameters for the model
            """
            with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdirname:
                cwd = os.getcwd()
                os.chdir(tmpdirname)
                input = set_input_method(z)
                res = subprocess.run(
                    command_args,
                    capture_output=True,
                    input=input,
                    text=True,
                    check=True,
                )
                output = get_output_method(res.stdout, res.stderr)
                os.chdir(cwd)
            return output

        return cls(model=model, **kwargs)

    @classmethod
    def from_model(
        cls, model: ForwardModelType, prior: Prior, fail_on_non_finite: bool = True
    ):
        """Convenience function to instantiate a Simulator with the correct sim_shapes.

        Args:
            model: simulator model.
            prior: model prior.
            fail_on_non_finite: whether return an invalid code if simulation
                returns NaN or infinite, default True

        Note:
            The simulator model is run once in order to infer observable shapes from the output.
        """
        obs = model(prior.sample(1)[0])
        sim_shapes = {k: v.shape for k, v in obs.items()}

        return cls(
            model=model, sim_shapes=sim_shapes, fail_on_non_finite=fail_on_non_finite
        )

    def set_dask_cluster(self, cluster=None) -> None:
        """Connect to Dask cluster.

        Args:
            cluster: cluster address or Cluster object from dask.distributed
                (default is LocalCluster)
        """
        if not (self.cluster is None and self.client is not None):
            self.client = Client(cluster)


def _run_model_chunk(
    z: np.ndarray,
    model: Callable,
    sim_shapes: SimShapeType,
    fail_on_non_finite: bool,
):
    """Run the model over a set of input parameters.

    Args:
        z: array with the input parameters. Should have shape (num. samples,
            num. parameters)
        model: simulator model function
        sim_shapes: map of simulator's output names to shapes
        fail_on_non_finite: whether return an invalid code if simulation
            returns NaN or infinite, default True
    Returns:
        x: dictionary with the output of the simulations
        status: array with the simulation status
    """
    chunk_size = len(z)
    x = {obs: np.full((chunk_size, *shp), np.nan) for obs, shp in sim_shapes.items()}
    status = np.zeros(len(z), dtype=np.int)
    for i, z_i in enumerate(z):
        out = model(z_i)
        _sim_stat = _get_sim_status(out, fail_on_non_finite)
        for obs, val in out.items():
            x[obs][i] = val
        status[i] = _sim_stat
    return x, status


def _get_sim_status(x: Mapping[str, Array], fail_on_non_finite: bool) -> int:
    """Did the simulation fail?"""

    assert isinstance(x, dict), "Simulators must return a dictionary."

    if any([v is None for v in x.values()]):
        return SimulationStatus.FAILED
    elif fail_on_non_finite and not all_finite(x):
        return SimulationStatus.FAILED
    else:
        return SimulationStatus.FINISHED
