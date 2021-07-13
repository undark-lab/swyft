# pylint: disable=no-member
from pathlib import Path
from typing import Dict, Hashable, Sequence, Tuple, Union

import numpy as np
import torch

# If you add a type, don't forget to update autodoc_type_aliases in /docs/source/config.py

PathType = Union[str, Path]
Device = Union[torch.device, str]
Tensor = torch.Tensor
Array = Union[np.ndarray, torch.Tensor]
Shape = Union[torch.Size, Tuple[int, ...]]

StrictPoIType = Tuple[int, ...]
PoIType = Union[Sequence[int], Sequence[StrictPoIType]]
MarginalType = Dict[PoIType, Array]

PNamesType = Sequence[str]
ObsType = Dict[Hashable, Array]
