from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from swyft.types import PathType


def depth(seq: Sequence):
    if isinstance(seq, (np.ndarray, torch.Tensor)):
        return seq.ndim
    elif seq and isinstance(seq, str):
        return 0
    elif seq and isinstance(seq, Sequence):
        return 1 + max(depth(item) for item in seq)
    else:
        return 0


def is_empty(directory: PathType):
    directory = Path(directory)
    if next(directory.iterdir(), None) is None:
        return True
    else:
        return False


if __name__ == "__main__":
    pass
