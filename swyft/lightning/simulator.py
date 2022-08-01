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
from tqdm import tqdm
from swyft.lightning.samples import *


###########
# Simulator
###########

def to_numpy(*args, single_precision = False):
    if len(args) > 1:
        result = []
        for arg in args:
            r = to_numpy(arg, single_precision = single_precision)
            result.append(r)
        return tuple(result)

    x = args[0]

    if isinstance(x, torch.Tensor):
        if not single_precision:
            return x.detach().cpu().numpy()
        else:
            x = x.detach().cpu()
            if x.dtype == torch.float64:
                x = x.float().numpy()
            else:
                x = x.numpy()
            return x
    elif isinstance(x, Samples):
        return Samples({k: to_numpy(v, single_precision = single_precision) for k, v in x.items()})
    elif isinstance(x, tuple):
        return tuple(to_numpy(v, single_precision = single_precision) for v in x)
    elif isinstance(x, list):
        return [to_numpy(v, single_precision = single_precision) for v in x]
    elif isinstance(x, dict):
        return {k: to_numpy(v, single_precision = single_precision) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        if not single_precision:
            return x
        else:
            if x.dtype == np.float64:
                x = np.float32(x)
            return x
    else:
        return x

def to_numpy32(*args):
    return to_numpy(*args, single_precision = True)
    
def to_torch(x):
    if isinstance(x, Samples):
        return Samples({k: to_torch(v) for k, v in x.items()})
    elif isinstance(x, dict):
        return {k: to_torch(v) for k, v in x.items()}
    else:
        return torch.as_tensor(x)
    

class Trace(dict):
    """Defines the computational graph (DAG) and keeps track of simulation results.
    """
    def __init__(self, targets = None, conditions = {}):
        """Instantiate Trace instante.

        Args:
            targets: Optional list of target sample variables. If provided, execution is stopped after those targets are evaluated. If `None`, all variables in the DAG will be evaluated.
            conditions: Optional `dict` or Callable. If a `dict`, sample variables will be conditioned to the corresponding values.  If Callable, it will be evaulated and it is expected to return a `dict`.
        """

        super().__init__(conditions)
        self._targets = targets
        self._prefix = ""

    def __repr__(self):
        return "Trace("+super().__repr__()+")"

    def __setitem__(self, k, v):
        if k not in self.keys():
            super().__setitem__(k, v)

    @property
    def covers_targets(self):
        return (self._targets is not None 
                and all([k in self.keys() for k in self._targets]))

    def sample(self, names, fn, *args, **kwargs):
        """Register sampling function.

        Args:
            names: Name or list of names of sampling variables.
            fn: Callable that returns the (list of) sampling variable(s).
            *args, **kwargs: Arguments and keywords arguments that are passed to `fn` upon evaluation.  LazyValues will be automatically evaluated if necessary.

        Returns:
            LazyValue sample.
        """
        assert callable(fn), "Second argument must be a function."
        if isinstance(names, list):
            names = [self._prefix + n for n in names]
            lazy_values = [LazyValue(self, k, names, fn, *args, **kwargs) for k in names]
            if self._targets is None or any([k in self._targets for k in names]):
                lazy_values[0].evaluate()
            return tuple(lazy_values)
        elif isinstance(names, str):
            name = self._prefix + names
            lazy_value = LazyValue(self, name, name, fn, *args, **kwargs)
            if self._targets is None or name in self._targets:
                lazy_value.evaluate()
            return lazy_value
        else:
            raise ValueError

    def prefix(self, prefix):
        return TracePrefixContextManager(self, prefix)


class TracePrefixContextManager:
    def __init__(self, trace, prefix):
        self._trace = trace
        self._prefix = prefix

    def __enter__(self):
        self._prefix, self._trace._prefix = self._trace._prefix, self._prefix + self._trace._prefix

    def __exit__(self, exception_type, exception_value, traceback):
        self._trace._prefix = self._prefix


class LazyValue:
    """Provides lazy evaluation functionality.
    """
    def __init__(self, trace, this_name, fn_out_names, fn, *args, **kwargs):
        """Instantiates LazyValue object.

        Args:
            trace: Trace instance (to be populated with sample).
            this_name: Name of this the variable that this LazyValue represents.
            fn_out_names: Name or list of names of variables that `fn` returns.
            fn: Callable that returns sample or list of samples.
            args, kwargs: Arguments and keyword arguments provided to `fn` upon evaluation.
        """
        self._trace = trace
        self._this_name = this_name
        self._fn_out_names = fn_out_names
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def __repr__(self):
        value = self._trace[self._this_name] if self._this_name in self._trace.keys() else "None"
        return f"LazyValue{self._this_name, value, self._fn, self._args, self._kwargs}"

    @property
    def value(self):
        """Value of this object."""
        return self.evaluate()

    def evaluate(self):
        """Trigger evaluation of function.

        Returns:
            Value of `this_name`.
        """
        if self._this_name not in self._trace.keys():
            args = (arg.evaluate() if isinstance(arg, LazyValue) else arg for arg in self._args)
            kwargs = {k: v.evaluate() if isinstance(v, LazyValue) else v for k, v in self._kwargs.items()}
            result = self._fn(*args, **kwargs)
            if not isinstance(self._fn_out_names, list):
                self._trace[self._fn_out_names] = result
            else:
                for out_name, value in zip(self._fn_out_names, result):
                    self._trace[out_name] = value
        return self._trace[self._this_name]


def collate_output(out):
    """Turn list of tensors/arrays-value dicts into dict of collated tensors or arrays"""
    keys = out[0].keys()
    result = {}
    for key in keys:
        if isinstance(out[0][key], torch.Tensor):
            result[key] = torch.stack([x[key] for x in out])
        else:
            result[key] = np.stack([x[key] for x in out])
    return result


class Simulator:
    """Handles simulations."""
    def on_before_forward(self, sample):
        """Apply transformations to conditions."""
        return sample

    @abstractmethod
    def forward(self, trace):
        """Main function to overwrite.
        """
        raise NotImplementedError

    def on_after_forward(self, sample):
        """Apply transformation to generated samples."""
        return sample

    def _run(self, targets = None, conditions = {}):
        conditions = conditions() if callable(conditions) else conditions

        conditions = self.on_before_forward(conditions)
        trace = Trace(targets, conditions)
        if not trace.covers_targets:
            self.forward(trace)
            #try:
            #    self.forward(trace)
            #except CoversTargetException:
            #    pass
        if targets is not None and not trace.covers_targets:
            raise ValueError("Missing simulation targets.")
        result = self.on_after_forward(dict(trace))

        return result
    
    def get_shapes_and_dtypes(self, targets = None):
        """Return shapes and data-types of sample variables.

        Args:
            targets: Target sample variables to simulate.

        Return:
            dictionary of shapes, dictionary of dtypes
        """
        sample = self(targets = targets)
        shapes = {k: tuple(v.shape) for k, v in sample.items()}
        dtypes = {k: v.dtype for k, v in sample.items()}
        return shapes, dtypes

    def __call__(self, trace):
        result = self.forward(trace)
        return result

    def sample(self, N = None, targets = None, conditions = {}, exclude = []):
        """Sample from the simulator.

        Args:
            N: int, number of samples to generate
            targets: Optional list of target sample variables to generate. If `None`, all targets are simulated.
            conditions: Dict or Callable, conditions sample variables.
            exclude: List of parameters that are excluded from the returned samples.
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

    def get_iterator(self, targets = None, conditions = {}):
        """Generates an iterator. Useful for iterative sampling.

        Args:
            targets: Optional list of target sample variables.
            conditions: Dict or Callable.
        """
        def iterator():
            while True:
                yield self._run(targets = targets, conditions = conditions)

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
        sims = self._simulator(conditions = conditions, targets = self._targets)
        return sims
