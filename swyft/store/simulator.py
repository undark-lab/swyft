import enum
import logging
import os
import shlex
import subprocess
import tempfile
import traceback
from operator import getitem
from typing import Callable, Mapping, Optional, Tuple, Union

import dask
import dask.array as da
import numpy as np
import zarr
from dask.distributed import Client, fire_and_forget

from swyft.prior import Prior, PriorTruncator
from swyft.types import ForwardModelType, ObsShapeType, ParameterNamesType, PathType
from swyft.utils import all_finite

log = logging.getLogger(__name__)


class SimulationStatus(enum.IntEnum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3


class Simulator:
    """Wrapper class for simulator.

    Args:
        model: Model function.
        parameter_names: List of parameter names, or number of parameters (interpreted
            as 'z0', 'z1', ...).
        sim_shapes: Dict describing model function output shapes.
        sim_dtype: Model output data type.
        fail_on_non_finite: whether return an invalid code if simulation
            returns NaN or infinite, default True

    Examples:

        >>> def model(v):
        >>>     mu = sum(v)  # mu = x + y + z
        >>>     nu = np.array([v[1], 2*v[2]])  # nu = [y, 2*z]
        >>>     return dict(mu=mu, nu=nu)
        >>> simulator = swyft.Simulator(model, ["x", "y", "z"], sim_shapes=dict(mu=(1,), nu=(2,))
    """

    def __init__(
        self,
        model: ForwardModelType,
        parameter_names: Union[ParameterNamesType, int],
        sim_shapes: ObsShapeType,
        sim_dtype: str = "f8",
        fail_on_non_finite: bool = True,
    ) -> None:
        self.model = model
        if isinstance(parameter_names, int):
            parameter_names = ["v%i" % i for i in range(parameter_names)]
        self.parameter_names = parameter_names
        self.sim_shapes = sim_shapes
        self.sim_dtype = sim_dtype
        self.fail_on_non_finite = fail_on_non_finite

    def _run(
        self,
        v: Union[zarr.Array, np.ndarray],
        sims: Mapping[str, Union[zarr.indexing.OIndex, np.ndarray]],
        sim_status: Union[zarr.indexing.OIndex, np.ndarray],
        indices: np.ndarray,
        **kwargs
    ) -> None:
        """Run the simulator on the input parameters.

        Args:
            v: Array-like object with all the input parameters. Should have
                shape (num. samples, num. parameters). Should have ``.shape``,
                ``.ndim``, ``.dtype``, and support numpy-style slicing.
            sims: Dictionary of array-like objects where to store the
                simulation output. All arrays should have the number of samples
                as the size of the first dimension. Arrays should support
                numpy-style setitem orthogonal indexing (e.g.
                ``array[:, [1, 2, 3]] = 0``).
            sim_status: Array-like object where to store the simulation status
                (size should be equal to the number of samples). It should
                support numpy-style setitem orthogonal indexing (e.g.
                ``array[:, [1, 2, 3]] = 0``).
            indices: Indices of the samples that need to be run by the
                simulator.
        """
        for i in indices:
            sim, status = _run_model(v[i], self.model, self.fail_on_non_finite)
            for k in sims.keys():
                sims[k][i] = sim.get(k, np.nan)
            sim_status[i] = status

    @classmethod
    def from_command(
        cls,
        command: str,
        parameter_names: Union[ParameterNamesType, int],
        sim_shapes: ObsShapeType,
        set_input_method: Callable,
        get_output_method: Callable,
        shell: bool = False,
        tmpdir: Optional[PathType] = None,
        sim_dtype: str = "f8",
    ):
        """Setup a simulator from a command line program.

        Args:
            command: Command-line program using shell-like syntax.
            set_input_method: Function to setup the simulator input. It should
                take one input argument (the array with the input parameters),
                and return any input to be passed to the program via stdin. If
                the simulator requires any input files to be present, this
                function should write these to disk.
            get_output_method: Function to retrieve results from the simulator
                output. It should take two input arguments (stdout and stderr
                of the simulator run) and return a dictionary with the
                simulator output shaped as described by the ``sim_shapes``
                argument. If the simulator writes output to disk, this function
                should parse the results from the file(s).
            shell: execute the specified command through the shell. NOTE: the
                following security considerations apply:
                https://docs.python.org/3/library/subprocess.html#security-considerations
            tmpdir: Root temporary directory where to run the simulator.
                Each instance of the simulator will run in a separate
                sub-folder. It must exist.
        """
        if shell:
            log.warning(
                "Your command-line program will run through a shell - check the "
                "following security considerations: "
                "https://docs.python.org/3/library/subprocess.html#security-considerations"
            )
        command_args = shlex.split(command) if not shell else command

        def model(v):
            """Closure to setup an instance of the simulator.

            Args:
                v: input parameter array.
            """
            with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdirname:
                cwd = os.getcwd()
                try:
                    os.chdir(tmpdirname)
                    input = set_input_method(v)
                    res = subprocess.run(
                        command_args,
                        capture_output=True,
                        input=input,
                        text=True,
                        check=True,
                        shell=shell,
                    )
                    output = get_output_method(res.stdout, res.stderr)
                finally:
                    os.chdir(cwd)
            return output

        return cls(
            model=model,
            parameter_names=parameter_names,
            sim_shapes=sim_shapes,
            sim_dtype=sim_dtype,
        )

    @classmethod
    def from_model(
        cls, model: ForwardModelType, prior: Prior, fail_on_non_finite: bool = True
    ):
        """Instantiate a Simulator with the correct sim_shapes.

        Args:
            model: Simulator model.
            prior: Model prior.

        Note:
            The simulator model is run once in order to infer observable shapes from the output.
        """
        v = PriorTruncator(prior, bound=None).sample(1)[0]
        sims = model(v)
        sim_shapes = {k: v.shape for k, v in sims.items()}
        dtype = [v.dtype.str for v in sims.values()][0]
        return cls(
            model=model,
            parameter_names=len(v),
            sim_shapes=sim_shapes,
            sim_dtype=dtype,
            fail_on_non_finite=fail_on_non_finite,
        )


class DaskSimulator(Simulator):
    """Setup and run the simulator engine, powered by dask."""

    client = None

    def _run(
        self,
        v: Union[zarr.Array, np.ndarray],
        sims: Mapping[str, Union[zarr.indexing.OIndex, np.ndarray]],
        sim_status: Union[zarr.indexing.OIndex, np.ndarray],
        indices: np.ndarray,
        collect_in_memory: bool = True,
        batch_size: Optional[int] = None,
    ) -> None:
        """Run the simulator on the input parameters.

        Args:
            collect_in_memory: if True, collect the simulation output in
                memory; if False, instruct Dask workers to save the output to
                the corresponding arrays. The latter option is asynchronous,
                thus this method immediately returns.
            batch_size: simulations will be submitted in batches of the
                specified size
        """
        if self.client is None:
            self.set_dask_cluster()

        # open parameter array as Dask array
        chunks = getattr(v, "chunks", "auto")
        v_dask = da.from_array(v, chunks=chunks)
        idx = da.from_array(indices, chunks=(batch_size or -1,))
        v_dask = v_dask[idx]

        v_dask = v_dask.persist()  # load the parameters in the distributed memory

        # block-wise run the model function on the parameter array
        out = da.map_blocks(
            _run_model_chunk,
            v_dask,
            model=self.model,
            sim_shapes=self.sim_shapes,
            fail_on_non_finite=self.fail_on_non_finite,
            drop_axis=1,
            dtype=object,
            meta=np.array((), dtype=object),
        )

        # split result dictionary and simulation status array
        results = out.map_blocks(
            getitem, 0, meta=np.array((), dtype=object), dtype=object
        )
        status = out.map_blocks(getitem, 1, meta=np.array((), dtype=int), dtype=int)

        # unpack array of dictionaries to dictionary of arrays
        result_dict = {}
        for obs, shape in self.sim_shapes.items():
            result_dict[obs] = results.map_blocks(
                getitem,
                obs,
                new_axis=[i + 1 for i in range(len(shape))],
                chunks=(v_dask.chunks[0], *shape),
                meta=np.array(()),
                dtype=float,
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
            # Avoid Dask graph optimization in store (might cause simulations to be rerun)
            with dask.config.set({"optimization.fuse.active": False}):
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
                    lambda x: np.zeros(x.shape[0], dtype=int),
                    chunks=(source.chunks[0],),
                    drop_axis=[i for i in range(1, source.ndim)],
                    meta=np.array((), dtype=int),
                    dtype=int,
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

    def set_dask_cluster(self, cluster=None) -> None:
        """Connect to Dask cluster.

        Args:
            cluster: Cluster address or Cluster object from dask.distributed
                (default is LocalCluster).
        """
        self.client = Client(cluster)


def _run_model(
    v: np.ndarray, model: Callable, fail_on_non_finite: bool
) -> Tuple[Mapping[str, np.ndarray], int]:
    """Run one instance of the model.

    Args:
        v: Array with the model input parameters.
        model: Simulator model function.
        fail_on_non_finite: Whether return an invalid code if simulation
            returns NaN or infinite, default True.

    Return:
        Dictionary with the output of the model run, simulation status code.
    """
    sim = {}
    sim_status = SimulationStatus.FAILED
    try:
        sim = model(v)
        sim_status = _get_sim_status(sim, fail_on_non_finite)
    except:
        traceback.print_exc()
    return sim, sim_status


def _run_model_chunk(
    v: np.ndarray, model: Callable, sim_shapes: ObsShapeType, fail_on_non_finite: bool
) -> Tuple[Mapping[str, np.ndarray], np.ndarray]:
    """Run the model over a set of input parameters.

    Args:
        v: Array with the input parameters. Should have shape (num. samples,
            num. parameters).
        model: Simulator model function.
        sim_shapes: Map of simulator's output names to shapes.
        fail_on_non_finite: Whether return an invalid code if simulation
            returns NaN or infinite, default True.
    Returns:
        Dictionary with the output of the simulations, array with the simulation status.
    """
    chunk_size = len(v)
    sims = {obs: np.full((chunk_size, *shp), np.nan) for obs, shp in sim_shapes.items()}
    sim_status = np.zeros(len(v), dtype=int)
    for i, z_i in enumerate(v):
        sim, status = _run_model(z_i, model, fail_on_non_finite)
        for key, val in sim.items():
            sims[key][i] = val
        sim_status[i] = status
    return sims, sim_status


def _get_sim_status(sims: Mapping[str, np.ndarray], fail_on_non_finite: bool) -> int:
    """Did the simulation fail?"""

    assert isinstance(sims, dict), "Simulators must return a dictionary."

    if any([v is None for v in sims.values()]):
        return SimulationStatus.FAILED
    elif fail_on_non_finite and not all_finite(sims):
        return SimulationStatus.FAILED
    else:
        return SimulationStatus.FINISHED
