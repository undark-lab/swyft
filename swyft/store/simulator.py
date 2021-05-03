import os
import shlex
import subprocess
import tempfile
import enum
from operator import getitem
from typing import Callable, List, Mapping, Optional, Tuple, Union

import dask.array as da
import numpy as np
from dask.distributed import Client, fire_and_forget, wait

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
            sim_shapes: TODO
            fail_on_non_finite: whether return an invalid code if simulation
                returns NaN or infinite, default True
            cluster: cluster address or Cluster object from dask.distributed
                (default is LocalCluster)
        """
        self.model = model
        self.sim_shapes = sim_shapes
        self.fail_on_non_finite = fail_on_non_finite
        self.set_dask_cluster(cluster)

    def run(
        self,
        z: Union[np.ndarray, da.Array],
        store,
        indices,
        batch_size: Optional[int] = None,
    ) -> Tuple:  # TODO specify tuple element tyoes
        """Run the simulator on the input parameters.

        Args:
            z: array of input parameters that need to be run by the simulator.
                Should have shape (num. samples, num. parameters)
            batch_size: simulations will be submitted in batches of the specified
                size

        Returns:
            # TODO
        """
        n_samples, n_parameters = z.shape

        # split parameter array in chunks corresponding to sample subsets
        z = da.array(z)
        z = da.rechunk(
            z,
            chunks=(batch_size or n_samples, n_parameters),
        )

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
        is_valid = out.map_blocks(getitem, 1, meta=np.array(()), dtype=np.bool)

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
        
        delayed = [
            is_valid.store(
                store.sim_status.oindex,
                regions=(indices.tolist(),),
                lock=False,
                compute=False,
            )
        ]

        for key, value in store.sims.items():
            task = result_dict[key].store(
                value.oindex, regions=(indices.tolist(),), lock=False, compute=False
            )
            delayed.append(task)
        
        futures = self.client.compute(delayed)
        fire_and_forget(futures)
        wait(futures)
        
        return

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
           #TODO


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
        cls, model: Callable, prior, fail_on_non_finite: bool = True
    ):  # TODO define type of prior
        """Convenience function to instantiate a Simulator with the correct sim_shapes.

        Args:
            model: simulator model.
            prior: model prior.

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


def _get_sim_status(x: Mapping[str, Array], fail_on_non_finite: bool) -> bool:
    """Did the simulation fail?"""

    assert isinstance(x, dict), "Simulators must return a dictionary."

    if any([v is None for v in x.values()]):
        return SimulationStatus.FAILED
    elif fail_on_non_finite and not all_finite(x):
        return SimulationStatus.FAILED
    else:
        return SimulationStatus.FINISHED
