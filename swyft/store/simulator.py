import enum
import os
import shlex
import subprocess
import tempfile
from operator import getitem
from typing import Callable, Mapping, Optional

import dask.array as da
import numpy as np
from dask.distributed import Client, fire_and_forget, wait

from swyft.bounds import Prior
from swyft.types import Array, PathType, Shape
from swyft.utils import all_finite


class SimulationStatus(enum.IntEnum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3


class Simulator:
    """ Setup and run the simulator engine """

    def __init__(
        self,
        model: Callable,
        sim_shapes: Mapping[str, Shape],
        fail_on_non_finite: bool = True,
        cluster=None,
    ):
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
        self.cluster = cluster

    def run(
        self,
        pars,
        sims,
        sim_status,
        indices,
        f_collect: bool = True,
        batch_size: Optional[int] = None,
        wait_for_results: bool = True,
    ):
        """Run the simulator on the input parameters.

        Args:
            pars: Zarr array with all the input parameters. Should have shape
                (num. samples, num. parameters)
            sims: Zarr group where to store all the simulation output
            sim_status: Zarr array where to store all the simulation status
            indices: indices of the samples that need to be run by the simulator
            f_collect: if True, collect all samples' output and pass this to the
                Zarr store; if False, instruct Dask workers to save output directly
                to the Zarr store
            batch_size: simulations will be submitted in batches of the specified
                size
            wait_for_results: if True, return only when all simulations are done
        """
        self.set_dask_cluster(self.cluster)

        # open parameter array as Dask array
        chunks = getattr(pars, "chunks", "auto")
        z = da.from_array(pars, chunks=chunks)
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

        if f_collect:
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
                    lambda x: 0,
                    chunks=(source.chunks[0],),
                    drop_axis=[i for i in range(1, source.ndim)],
                    meta=np.array((), dtype=np.int),
                    dtype=np.int,
                )
                for source in sources
            ]
            status = da.add(*zeros_when_done, status)
            status = status.store(
                target=sim_status,
                regions=(indices.tolist(),),
                lock=False,
                compute=False,
            )
            # when the simulation results are stored, we can update the status
            status_stored = self.client.compute(status)
            fire_and_forget(status_stored)

            if wait_for_results:
                wait(status_stored)

    @classmethod
    def from_command(
        cls,
        command: str,
        set_input_method: Callable,
        get_output_method: Callable,
        tmpdir: PathType = None,
        **kwargs,
    ):
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
    def from_model(cls, model: Callable, prior: Prior, fail_on_non_finite: bool = True):
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

    def set_dask_cluster(self, cluster=None) -> None:  # TODO type for cluster
        """Connect to Dask cluster.

        Args:
            cluster: cluster address or Cluster object from dask.distributed
                (default is LocalCluster)
        """
        self.client = Client(cluster)


def _run_model_chunk(
    z: np.ndarray,
    model: Callable,
    sim_shapes: Mapping[str, Shape],
    fail_on_non_finite: bool,
):
    """Run the model over a set of input parameters.

    Args:
        # TODO

    Returns:
        # TODO
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
