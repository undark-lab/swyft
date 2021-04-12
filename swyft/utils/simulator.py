# New
import os
import shlex
import subprocess
import tempfile
from operator import getitem
from typing import Callable, List, Mapping, Optional, Tuple, Union

import dask.array as da
import numpy as np
from dask.distributed import Client

from swyft.types import Array, PathType, Shape
from swyft.utils import all_finite


class Simulator:
    """ Setup and run the simulator engine """

    def __init__(
        self,
        model: Callable,
        params: List,
        obs_shapes: Mapping[str, Shape],
        fail_on_non_finite: bool = True,
    ):
        """Initiate Simulator using a python function.

        Args:
            model: simulator model function
            params: TODO
            obs_shapes: TODO
            fail_on_non_finite: whether return an invalid code if simulation
                returns NaN or infinite, default True
        """
        self.model = model
        self.params = params
        self.obs_shapes = obs_shapes
        self.fail_on_non_finite = fail_on_non_finite

        self.client = None

    def set_dask_cluster(self, cluster=None) -> None:  # TODO type for cluster
        """Connect to Dask cluster.

        Args:
            cluster: cluster address or Cluster object from dask.distributed
                (default is LocalCluster)
        """
        self.client = Client(cluster)

    def run(
        self,
        z: Union[np.ndarray, da.Array],
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
        assert n_parameters == len(self.params), "Wrong number of parameters"

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
            params=self.params,
            obs_shapes=self.obs_shapes,
            fail_on_non_finite=self.fail_on_non_finite,
            drop_axis=1,
            dtype=np.object,
        )

        # split result dictionary and simulation status array
        results = out.map_blocks(getitem, 0, dtype=np.object)
        is_valid = out.map_blocks(getitem, 1, meta=np.array(()), dtype=np.bool)

        # unpack array of dictionaries to dictionary of arrays
        result_dict = {}
        for obs, shape in self.obs_shapes.items():
            result_dict[obs] = results.map_blocks(
                getitem,
                obs,
                new_axis=[i + 1 for i in range(len(shape))],
                chunks=(z.chunks[0], *shape),
                meta=np.array(()),
                dtype=np.float,
            )
        return result_dict, is_valid

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
            get_output_method: method to retrieve results from the simulator
                output
            tmpdir: temporary directory where to run the simulator instances
                (one in each subdir). tmpdir must exist.
            **kwargs: other key-word arguments required to initialize the
                Simulator object.
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
    def from_model(cls, model: Callable, prior):  # TODO define type of prior
        """Convenience function to instantiate a Simulator with the correct obs_shapes.

        Args:
            model: simulator model.
            prior: model prior.

        Note:
            The simulator model is run once in order to infer observable shapes from the output.
        """
        params = prior.sample(1)
        params = {k: v.item() for k, v in params.items()}
        obs = model(params)
        obs_shapes = {k: v.shape for k, v in obs.items()}

        return cls(
            model=model, params=list(prior.prior_config.keys()), obs_shapes=obs_shapes
        )


def _run_model_chunk(
    z: np.ndarray,
    model: Callable,
    params: List,
    obs_shapes: Mapping[str, Shape],
    fail_on_non_finite: bool,
):
    """Run the model over a set of input parameters.

    Args:
        # TODO

    Returns:
        # TODO
    """
    chunk_size = len(z)
    x = {obs: np.full((chunk_size, *shp), np.nan) for obs, shp in obs_shapes.items()}
    has_failed = np.zeros(len(z), dtype=np.bool)
    for i, z_i in enumerate(z):
        inp = {p: val for p, val in zip(params, z_i)}
        out = model(inp)
        _failed = _has_failed(out, fail_on_non_finite)
        if _failed:
            for obs, val in out.items():
                x[obs][i] = val
            has_failed[i] = _failed
    return x, has_failed


def _has_failed(x: Mapping[str, Array], fail_on_non_finite: bool) -> bool:
    """Did the simulation fail?"""

    assert isinstance(x, dict), "Simulators must return a dictionary."

    if any([v is None for v in x.values()]):
        return True
    elif fail_on_non_finite and not all_finite(x):
        return True
    else:
        return False
