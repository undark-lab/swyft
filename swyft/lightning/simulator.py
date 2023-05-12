from abc import abstractmethod
from typing import (
    Callable,
    Dict,
    Hashable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import numpy as np
import torch
from tqdm.auto import tqdm
import swyft
import swyft.lightning.data
from swyft.lightning.data import *
from swyft.lightning.utils import collate_output


#########
# Samples
#########


class Sample(dict):
    """In Swyft, a 'sample' is a dictionary
    with string-type keys and tensor/array-type values."""

    def __repr__(self):
        return "Sample(" + super().__repr__() + ")"


class Samples(dict):
    """Handles memory-based samples in Swyft.  Samples are stored as dictionary
    of arrays/tensors with number of samples as first dimension. This class
    provides a few convenience methods for accessing the samples."""

    def __len__(self):
        """Number of samples."""
        n = [len(v) for v in self.values()]
        assert all([x == n[0] for x in n]), "Inconsistent lengths in Samples"
        return n[0]

    def __repr__(self):
        return "Samples(" + super().__repr__() + ")"

    def __getitem__(self, i):
        """For integers, return 'rows', for string returns 'columns'."""
        if isinstance(i, int):
            return Sample({k: v[i] for k, v in self.items()})
        elif isinstance(i, slice):
            return Samples({k: v[i] for k, v in self.items()})
        else:
            return super().__getitem__(i)

    def get_dataset(self, on_after_load_sample=None):
        """Generator function for SamplesDataset object.

        Args:
            on_after_load_sample: Callable, that is applied to individual samples on the fly.

        Returns:
            SamplesDataset
        """
        return swyft.lightning.data.SamplesDataset(
            self, on_after_load_sample=on_after_load_sample
        )

    def get_dataloader(
        self,
        batch_size=1,
        shuffle=False,
        on_after_load_sample=None,
        repeat=None,
        num_workers=0,
    ):
        """Generator function to directly generate a dataloader object.

        Args:
            batch_size: batch_size for dataloader
            shuffle: shuffle for dataloader
            on_after_load_sample: see `get_dataset`
            repeat: If not None, Wrap dataset in RepeatDatasetWrapper
        """
        dataset = self.get_dataset(on_after_load_sample=on_after_load_sample)
        if repeat is not None:
            dataset = swyft.lightning.data.RepeatDatasetWrapper(dataset, repeat=repeat)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )


#######
# Graph
#######


class Node:
    """Provides lazy evaluation functionality."""

    def __init__(self, parname, mult_parnames, fn, *inputs):
        """Instantiates LazyValue object.

        Args:
            trace: Trace instance (to be populated with sample).
            this_name: Name of the variable that this LazyValue represents.
            fn_out_names: Name or list of names of variables that `fn` returns.
            fn: Callable that returns sample or list of samples.
            args, kwargs: Arguments and keyword arguments provided to `fn` upon evaluation.
        """
        self._parname = parname
        self._mult_parnames = mult_parnames
        self._fn = fn
        self._inputs = inputs

    def __repr__(self):
        return f"Node{self._parname, self._fn, self._inputs}"

    def evaluate(self, trace):
        if self._parname in trace.keys():  # Nothing to do
            return trace[self._parname]
        else:
            args = (
                arg.evaluate(trace)
                if (isinstance(arg, Node) or isinstance(arg, Switch))
                else arg
                for arg in self._inputs
            )
            result = self._fn(*args)
            if self._mult_parnames is None:
                trace[self._parname] = result
            else:
                for parname, value in zip(self._mult_parnames, result):
                    trace[parname] = value
            return trace[self._parname]


class Switch:
    """Provides lazy evaluation functionality."""

    def __init__(self, parname, options, choice):
        self._parname = parname
        self._options = options
        self._choice = choice

    def evaluate(self, trace):
        if self._parname in trace.keys():  # Nothing to do
            return trace[self._parname]
        else:
            choice = self._choice.evaluate(trace)
            choice = int(choice)  # type-cast if possible
            result = self._options[choice].evaluate(trace)
            trace[self._parname] = result
            return result


class Graph:
    """Defines the computational graph (DAG) and keeps track of simulation results."""

    def __init__(self):
        self.nodes = {}
        self._prefix = ""

    def __repr__(self):
        return "Graph(" + self.nodes.__repr__() + ")"

    def __setitem__(self, key, value):
        if key not in self.nodes.keys():
            self.nodes.__setitem__(key, value)

    def keys(self):
        return self.nodes.keys()

    def __getitem__(self, key):
        return self.nodes[key]

    def node(self, parnames, fn, *args):
        """Register sampling function.

        Args:
            parnames: Name or list of names of sampling variables.
            fn: Callable that returns the (list of) sampling variable(s).
            *args: Arguments and keywords arguments that are passed to `fn` upon evaluation.  LazyValues will be automatically evaluated if necessary.

        Returns:
            Node or tuple of nodes.
        """
        assert callable(fn), "Second argument must be a function."
        if isinstance(parnames, str):
            parnames = self._prefix + parnames
            node = Node(parnames, None, fn, *args)
            self.nodes[parnames] = node
            return node
        else:
            parnames = [self._prefix + n for n in parnames]
            nodes = tuple(Node(parname, parnames, fn, *args) for parname in parnames)
            for i, parname in enumerate(parnames):
                self.nodes[parname] = nodes[i]
            return nodes

    def switch(self, parname, options, choice):
        switch = Switch(parname, options, choice)
        self.nodes[parname] = switch
        return switch

    def prefix(self, prefix):
        return GraphPrefixContextManager(self, prefix)


class GraphPrefixContextManager:
    def __init__(self, graph, prefix):
        self._graph = graph
        self._prefix = prefix

    def __enter__(self):
        self._prefix, self._graph._prefix = (
            self._graph._prefix,
            self._prefix + self._graph._prefix,
        )

    def __exit__(self, exception_type, exception_value, traceback):
        self._graph._prefix = self._prefix


###########
# Simulator
###########


class Simulator:
    r"""Base class for defining a simulator in Swyft.

    This class provides a framework for the definition of the computational graph of the simulation model, and methods for its efficient execution. The computational graph is build in terms of labeled notes in the `build' method. This method is only ev


    Example usage:

    .. code-block:: python

       class MySim(swyft.Simulator):
           def __init__(self):
               super().__init__()
               self.transform_samples = swyft.to_numpy32

           def build(self, graph):
               z = graph.node('z', lambda: np.random.rand(1))
               x = graph.node('x', lambda z: z + np.random.randn(1)*0.1, z)
    """

    def __init__(self):
        self.graph = None

    #        self.build_graph(self.graph)

    def transform_conditions(self, conditions):
        return conditions

    def build(self, graph: Graph):
        """To be overwritten in derived classes (see example usage above).

        .. note::

           This method only runs *once* after Simulator instantiation, during the generation of the very first
           sample. Afterwards, the graph object itself (and the
           functions its nodes point to) is used for performing computations.

        Args:
            graph: Graph object instance, to be populated with nodes during method execution.

        Returns:
            None
        """
        raise NotImplementedError("Missing!")

    def transform_samples(self, sample: Sample):
        """Hook for applying transformation to generated samples.  Should be overwritten by user.

        A typical use-case is to change the data-type of the samples to single precision (if applicable).  Swyft provides some convenience functions that can be used in this case. See above `to_numpy32' for an example.

        Args:
            sample: Input sample

        Returns:
            Sample: Transformed sample.
        """
        return sample

    def _run(self, targets=None, conditions={}):
        if self.graph is None:
            self.graph = Graph()
            self.build(self.graph)
        conditions = conditions() if callable(conditions) else conditions
        conditions = self.transform_conditions(conditions)
        trace = dict(conditions)
        if targets is None:
            targets = self.graph.keys()
        for target in targets:
            self.graph[target].evaluate(trace)
        result = self.transform_samples(trace)
        return result

    def get_shapes_and_dtypes(self, targets: Optional[Sequence[str]] = None):
        """This function run the simulator once and collects information about
        shapes and data-types of the nodes of the computational graph.

        Args:
            targets: Optional list of target sample variables.  If None, the full simulation model is run.

        Return:
            (Dict, Dict): Dictionary of shapes and dictionary of dtypes
        """
        sample = self.sample(targets=targets)
        shapes = {k: tuple(v.shape) for k, v in sample.items()}
        dtypes = {k: v.dtype for k, v in sample.items()}
        return shapes, dtypes

    def sample(
        self,
        N: Optional[int] = None,
        targets: Optional[Sequence[str]] = None,
        conditions: Union[Dict, Callable] = {},
        exclude: Optional[Sequence[str]] = [],
    ):
        """Sample from the simulator.

        Args:
            N: Number of samples to generate.  If None, a single sample without sample dimension is returned.
            targets: Optional list of target sample variables to generate. If `None`, all targets are simulated.
            conditions: Dict or Callable, conditions on sample variables.  A
                callable will be executed separately for each sample and is expected to return
                a dictionary with conditions.
            exclude: Optional list of parameters that are excluded from the
                returned samples.  Can be used to reduce memory consumption.
        """
        if N is None:
            return Sample(self._run(targets, conditions))

        out = []
        for _ in tqdm(range(N)):
            result = self._run(targets, conditions)
            for key in exclude:
                result.pop(key, None)
            out.append(result)
        out = collate_output(out)
        out = Samples(out)
        return out

    def get_resampler(self, targets):
        """Generates a resampler. Useful for noise hooks etc.

        Args:
            targets: List of target variables to simulate

        Returns:
            SimulatorResampler instance.
        """
        return SimulatorResampler(self, targets)

    def get_iterator(self, targets=None, conditions={}):
        """Generates an iterator. Useful for iterative sampling.

        Args:
            targets: Optional list of target sample variables.
            conditions: Dict or Callable.
        """

        def iterator():
            while True:
                yield self._run(targets=targets, conditions=conditions)

        return iterator


class SimulatorResampler:
    """Handles rerunning part of the simulator. Typically used for on-the-fly calculations during training."""

    def __init__(self, simulator, targets):
        """Instantiates SimulatorResampler

        Args:
            simulator: The simulator object
            targets: List of target sample variables that will be resampled
        """
        self._simulator = simulator
        self._targets = targets

    def __call__(self, sample):
        """Resamples.

        Args:
            sample: Sample dict

        Returns:
            sample with resampled sites
        """
        conditions = sample.copy()
        for k in self._targets:
            conditions.pop(k)
        sims = self._simulator.sample(conditions=conditions, targets=self._targets)
        return sims


# class Trace(dict):
#    """Defines the computational graph (DAG) and keeps track of simulation results."""
#
#    def __init__(self, targets=None, conditions={}):
#        """Instantiate Trace instante.
#
#        Args:
#            targets: Optional list of target sample variables. If provided, execution is stopped after those targets are evaluated. If `None`, all variables in the DAG will be evaluated.
#            conditions: Optional `dict` or Callable. If a `dict`, sample variables will be conditioned to the corresponding values.  If Callable, it will be evaulated and it is expected to return a `dict`.
#        """
#
#        super().__init__(conditions)
#        self._targets = targets
#        self._prefix = ""
#
#    def __repr__(self):
#        return "Trace(" + super().__repr__() + ")"
#
#    def __setitem__(self, k, v):
#        if k not in self.keys():
#            super().__setitem__(k, v)
#
#    @property
#    def covers_targets(self):
#        return self._targets is not None and all(
#            [k in self.keys() for k in self._targets]
#        )
#
#    def sample(self, names, fn, *args, **kwargs):
#        """Register sampling function.
#
#        Args:
#            names: Name or list of names of sampling variables.
#            fn: Callable that returns the (list of) sampling variable(s).
#            *args, **kwargs: Arguments and keywords arguments that are passed to `fn` upon evaluation.  LazyValues will be automatically evaluated if necessary.
#
#        Returns:
#            LazyValue sample.
#        """
#        assert callable(fn), "Second argument must be a function."
#        if isinstance(names, list):
#            names = [self._prefix + n for n in names]
#            lazy_values = [
#                LazyValue(self, k, names, fn, *args, **kwargs) for k in names
#            ]
#            if self._targets is None or any([k in self._targets for k in names]):
#                lazy_values[0].evaluate()
#            return tuple(lazy_values)
#        elif isinstance(names, str):
#            name = self._prefix + names
#            lazy_value = LazyValue(self, name, name, fn, *args, **kwargs)
#            if self._targets is None or name in self._targets:
#                lazy_value.evaluate()
#            return lazy_value
#        else:
#            raise ValueError
#
#    def prefix(self, prefix):
#        return TracePrefixContextManager(self, prefix)
#
#
# class TracePrefixContextManager:
#    def __init__(self, trace, prefix):
#        self._trace = trace
#        self._prefix = prefix
#
#    def __enter__(self):
#        self._prefix, self._trace._prefix = (
#            self._trace._prefix,
#            self._prefix + self._trace._prefix,
#        )
#
#    def __exit__(self, exception_type, exception_value, traceback):
#        self._trace._prefix = self._prefix
#
#
# class LazyValue:
#    """Provides lazy evaluation functionality."""
#
#    def __init__(self, trace, this_name, fn_out_names, fn, *args, **kwargs):
#        """Instantiates LazyValue object.
#
#        Args:
#            trace: Trace instance (to be populated with sample).
#            this_name: Name of this the variable that this LazyValue represents.
#            fn_out_names: Name or list of names of variables that `fn` returns.
#            fn: Callable that returns sample or list of samples.
#            args, kwargs: Arguments and keyword arguments provided to `fn` upon evaluation.
#        """
#        self._trace = trace
#        self._this_name = this_name
#        self._fn_out_names = fn_out_names
#        self._fn = fn
#        self._args = args
#        self._kwargs = kwargs
#
#    def __repr__(self):
#        value = (
#            self._trace[self._this_name]
#            if self._this_name in self._trace.keys()
#            else "None"
#        )
#        return f"LazyValue{self._this_name, value, self._fn, self._args, self._kwargs}"
#
#    @property
#    def value(self):
#        """Value of this object."""
#        return self.evaluate()
#
#    def evaluate(self):
#        """Trigger evaluation of function.
#
#        Returns:
#            Value of `this_name`.
#        """
#        if self._this_name not in self._trace.keys():
#            args = (
#                arg.evaluate() if isinstance(arg, LazyValue) else arg
#                for arg in self._args
#            )
#            kwargs = {
#                k: v.evaluate() if isinstance(v, LazyValue) else v
#                for k, v in self._kwargs.items()
#            }
#            result = self._fn(*args, **kwargs)
#            if not isinstance(self._fn_out_names, list):
#                self._trace[self._fn_out_names] = result
#            else:
#                for out_name, value in zip(self._fn_out_names, result):
#                    self._trace[out_name] = value
#        return self._trace[self._this_name]


# class SimulatorOld:
#    """Handles simulations."""
#
#    def on_before_forward(self, sample):
#        """Apply transformations to conditions.
#
#        DEPRECATED: Use `transform_conditions` instead
#        """
#        return sample
#
#    def transform_conditions(self, conditions):
#        return conditions
#
#    @abstractmethod
#    def forward(self, trace):
#        """Main function to overwrite."""
#        raise NotImplementedError
#
#    def on_after_forward(self, sample):
#        """Apply transformation to generated samples.
#
#        DEPRECATEDE: Use `transform_samples` instead
#        """
#        return sample
#
#    def transform_samples(self, sample):
#        """Apply transformation to generated samples."""
#        return sample
#
#    def _run(self, targets=None, conditions={}):
#        conditions = conditions() if callable(conditions) else conditions
#
#        conditions = self.on_before_forward(conditions)
#        conditions = self.transform_conditions(conditions)
#        trace = Trace(targets, conditions)
#        if not trace.covers_targets:
#            self.forward(trace)
#            # try:
#            #    self.forward(trace)
#            # except CoversTargetException:
#            #    pass
#        if targets is not None and not trace.covers_targets:
#            raise ValueError("Missing simulation targets.")
#        result = self.on_after_forward(dict(trace))
#        result = self.transform_samples(result)
#
#        return result
#
#    def get_shapes_and_dtypes(self, targets=None):
#        """Return shapes and data-types of sample variables.
#
#        Args:
#            targets: Target sample variables to simulate.
#
#        Return:
#            dictionary of shapes, dictionary of dtypes
#        """
#        sample = self.sample(targets=targets)
#        shapes = {k: tuple(v.shape) for k, v in sample.items()}
#        dtypes = {k: v.dtype for k, v in sample.items()}
#        return shapes, dtypes
#
#    def __call__(self, trace):
#        result = self.forward(trace)
#        return result
#
#    def sample(self, N=None, targets=None, conditions={}, exclude=[]):
#        """Sample from the simulator.
#
#        Args:
#            N: int, number of samples to generate
#            targets: Optional list of target sample variables to generate. If `None`, all targets are simulated.
#            conditions: Dict or Callable, conditions sample variables.
#            exclude: List of parameters that are excluded from the returned samples.
#        """
#        if N is None:
#            return Sample(self._run(targets, conditions))
#
#        out = []
#        for _ in tqdm(range(N)):
#            result = self._run(targets, conditions)
#            for key in exclude:
#                result.pop(key, None)
#            out.append(result)
#        out = collate_output(out)
#        out = Samples(out)
#        return out
#
#    def get_resampler(self, targets):
#        """Generates a resampler. Useful for noise hooks etc.
#
#        Args:
#            targets: List of target variables to simulate
#
#        Returns:
#            SimulatorResampler instance.
#        """
#        return SimulatorResampler(self, targets)
#
#    def get_iterator(self, targets=None, conditions={}):
#        """Generates an iterator. Useful for iterative sampling.
#
#        Args:
#            targets: Optional list of target sample variables.
#            conditions: Dict or Callable.
#        """
#
#        def iterator():
#            while True:
#                yield self._run(targets=targets, conditions=conditions)
#
#        return iterator
