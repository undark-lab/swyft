# New
import os
import shlex
import subprocess
import tempfile
from typing import Dict

import dask.bag as db
from dask.distributed import Client

from swyft.types import Array
from swyft.utils import all_finite


class Simulator:
    """ Setup and run the simulator engine """

    def __init__(self, model, sim_shapes, fail_on_non_finite: bool = True):
        """
        initialte Simulator

        Args:
            model (callable): simulator model function
            fail_on_non_finite (bool): whether return a invalid code if simulation returns infinite, default True
        """
        self.model = model
        self.client = None
        self.sim_shapes = sim_shapes
        self.fail_on_non_finite = fail_on_non_finite

    def set_dask_cluster(self, cluster=None):
        """
        Connect to Dask cluster

        Args:
            cluster (string or Cluster): cluster address or Cluster object
                                         from dask.distributed (default is
                                         LocalCluster)
        """
        self.client = Client(cluster)

    def run(self, z, npartitions=None):
        """
        Run the simulator on the input parameters

        Args:
            z (list of dictionary): set of input parameters that need to
                                    be run by the simulator
            npartitions (int): number of partitions in which the input
                               parameters are divided for the parallelization
                               (default is about 100)
        """

        print("Simulator: Running...")
        bag = db.from_sequence(z, npartitions=npartitions)
        bag = bag.map(_run_one_sample, self.model, self.fail_on_non_finite)
        result = bag.compute(scheduler=self.client or "processes")
        print("Simulator: ...done.")
        return result

    @classmethod
    def from_command(cls, command, set_input_method, get_output_method, tmpdir=None):
        """
        Setup command-line simulator

        Args:
            command (string): command line simulator
            set_input_method (callable): method to prepare simulator input
            get_output_method (callable): method to retrieve results from the
                                          simulator output
            tmpdir (string): temporary directory where to run the simulator
                             instances (one in each subdir). tmpdir must exist.
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

        return cls(model=model)


def _succeed(x: Dict[str, Array], fail_on_non_finite: bool) -> int:
    """Is the simulation a success?"""

    # Code disctionary for validity
    code = {"valid": 0, "none_value": 1, "non_finite_value": 2}

    assert isinstance(x, dict), "Simulators must return a dictionary."

    if any([v is None for v in x.values()]):
        return code["none_value"]
    elif fail_on_non_finite and not all_finite(x):
        return code["non_finite_value"]
    else:
        return code["valid"]


def _run_one_sample(param, model, fail_on_non_finite):
    """Run model for one set of parameters and check validity of the output.

    Args:
        param (dictionary): one set of input parameters
        model (callable): simulator model
        fail_on_non_finite (bool): whether return a invalid code if simulation
            returns infinite, default True
    """
    x = model(param)
    validity = _succeed(x, fail_on_non_finite)
    return (x, validity)
