import dask.bag as db
import os
import shlex
import subprocess
import tempfile

from dask.distributed import Client


class Simulator:
    """ Setup and run the simulator engine """
    def __init__(self, simulator):
        self.simulator = simulator

        self.client = None

    def set_dask_cluster(self, cluster=None):
        """
        Connect to Dask cluster

        Args:
            cluster (string or Cluster): cluster address or Cluster object
                                         from dask.distributed (default is
                                         LocalCluster)
        """
        self.client = Client(cluster)

    def run(self, input_parameters, npartitions=None):
        """
        Run the simulator on the input parameters

        Args:
            input_parameters (iterable): set of input parameters that need to
                                         be run by the simulator
            npartitions (int): number of partitions in which the input
                               parameters are divided for the parallelization
                               (default is about 100)
        """
        bag = db.from_sequence(input_parameters, npartitions=npartitions)
        bag = bag.map(self.simulator)
        return bag.compute(scheduler=self.client or 'processes')

    @classmethod
    def from_command(cls, command, set_input_method, get_output_method,
                     tmpdir=None):
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

        def simulator(z):
            """
            Closure to run an instance of the simulator

            Args:
                z (array-like): input parameters for the simulator
            """
            with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdirname:
                cwd = os.getcwd()
                os.chdir(tmpdirname)
                input = set_input_method(z)
                res = subprocess.run(command_args,
                                     capture_output=True,
                                     input=input,
                                     text=True,
                                     check=True)
                output = get_output_method(res.stdout, res.stderr)
                os.chdir(cwd)
            return output

        return cls(simulator=simulator)
