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
from pathlib import Path
from collections.abc import Collection

import numpy as np
import torch

PathType = Union[str, Path]

Device = Union[torch.device, str]
Dataset = torch.utils.data.Dataset

Tensor = torch.Tensor
Array = Union[np.ndarray, torch.Tensor]

Combinations = Union[int, Sequence[int], Sequence[Sequence[int]]]

Shape = Union[torch.Size, Tuple[int, ...]]
DictInt = Union[int, Dict[str, int]]
DictShape = Union[Shape, Dict[str, Shape]]
