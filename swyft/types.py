from numbers import Real
from pathlib import Path
from typing import Callable, Dict, Hashable, Mapping, Sequence, Tuple, Union

import numpy as np
import torch

# from pandas import DataFrame

# If you add a type, don't forget to update autodoc_type_aliases in /docs/source/config.py

# General defintions
PathType = Union[str, Path]
Device = Union[torch.device, str]
Tensor = torch.Tensor
Array = Union[np.ndarray, torch.Tensor]
Shape = Union[torch.Size, Tuple[int, ...]]

# A list of marginals, e.g., [0, (3, 4), [3, 2]]
MarginalIndex = Union[int, Sequence[int], Sequence[Sequence[int]]]
# Strict version of that type used internally, ((0,), (2, 3), (3, 4))
StrictMarginalIndex = Tuple[Tuple[int, ...], ...]

# Map from (3, 4) --> array
MarginalToArray = Dict[Tuple[int, ...], Array]
# Map from (3, 4) --> DataFrame
# MarginalToDataFrame = Dict[Tuple[int, ...], DataFrame]

ParameterNamesType = Sequence[str]
ObsType = Dict[Hashable, Array]
ObsShapeType = Mapping[Hashable, Shape]
ForwardModelType = Callable[..., ObsType]

# plotting
BaseLimitType = Tuple[Real, Real]
LimitType = Union[BaseLimitType, Sequence[BaseLimitType]]
