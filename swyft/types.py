# pylint: disable=no-member
from pathlib import Path
from typing import Callable, Dict, Hashable, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np
import torch

# If you add a type, don't forget to update autodoc_type_aliases in /docs/source/config.py

# General defintions
PathType = Union[str, Path]
Device = Union[torch.device, str]
Tensor = torch.Tensor
Array = Union[np.ndarray, torch.Tensor]
Shape = Union[torch.Size, Tuple[int, ...]]

# A list of marginals, e.g., [0, (3, 4), [3, 2]]
MarginalsType = Iterable[Union[int, Iterable[int]]]

# Strict version of that type used internally, ((0,), (2, 3), (3, 4))
StrictMarginalsType = Tuple[Tuple[int, ...], ...]

# Map from (3, 4) --> ratio array
RatiosType = Dict[Tuple[int, ...], Array]

PNamesType = Sequence[str]
ObsType = Dict[Hashable, Array]
ForwardModelType = Callable[..., ObsType]
SimShapeType = Mapping[Hashable, Shape]

# Maybe obsolete?
StrictPoIType = Tuple[int, ...]
PoIType = Union[Sequence[int], Sequence[StrictPoIType]]
MarginalType = Dict[PoIType, Array]
