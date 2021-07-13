# pylint: disable=no-member
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch

# If you add a type, don't forget to update autodoc_type_aliases in /docs/source/config.py

PathType = Union[str, Path]
Device = Union[torch.device, str]
Tensor = torch.Tensor
Array = Union[np.ndarray, torch.Tensor]
Shape = Union[torch.Size, Tuple[int, ...]]

PoIType = Tuple[int, ...]
MarginalType = Dict[PoIType, Array]
