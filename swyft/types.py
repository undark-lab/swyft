# pylint: disable=no-member
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NewType,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from collections.abc import Collection

import numpy as np
import torch

Device = Union[torch.device, str]

Shape = Union[torch.Size, Tuple[int, ...]]

DictInt = Union[int, Dict[str, int]]
DictShape = Union[ Shape, Dict[str, Shape] ]

Tensor = torch.Tensor
Array = Union[np.ndarray, torch.Tensor]