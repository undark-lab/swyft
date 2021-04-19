# pylint: disable=no-member
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch

PathType = Union[str, Path]

Device = Union[torch.device, str]
Dataset = torch.utils.data.Dataset

Tensor = torch.Tensor
Array = Union[np.ndarray, torch.Tensor]

WeightsType = Dict[Tuple[int], np.ndarray]

Combinations = Union[int, Sequence[int], Sequence[Sequence[int]]]

Shape = Union[torch.Size, Tuple[int, ...]]
DictInt = Union[int, Dict[str, int]]
DictShape = Union[Shape, Dict[str, Shape]]

PriorConfig = Dict[str, Tuple[str, float, float]]
