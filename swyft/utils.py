# pylint: disable=no-member, not-callable
from warnings import warn
from pathlib import Path

import numpy as np
import torch

from .types import (
    Optional,
    Device,
    Tensor,
    Array,
    List,
    Sequence,
    Combinations,
    PathType,
)


def comb2d(indices):
    output = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            output.append([indices[i], indices[j]])
    return output


def combine_z(z: Tensor, combinations: Optional[List]) -> Tensor:
    """Generate parameter combinations in last dimension using fancy indexing.

    Args:
        z: parameters of shape [..., Z]
        combinations: list of parameter combinations.

    Returns:
        output = z[..., combinations]
    """
    return z[..., combinations]


def set_device(gpu: bool = False) -> torch.device:
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    elif gpu and not torch.cuda.is_available():
        warn("Although the gpu flag was true, the gpu is not avaliable.")
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    return device


def get_x(list_xz):
    """Extract x from batch of samples."""
    return [xz["x"] for xz in list_xz]


def get_z(list_xz):
    """Extract z from batch of samples."""
    return [xz["z"] for xz in list_xz]


def get_device_if_not_none(device: Optional[Device], tensor: Tensor) -> Device:
    """Returns device if not None, else returns tensor.device."""
    return tensor.device if device is None else device


np_bool_types = [np.bool]
np_int_types = [np.int8, np.int16, np.int32, np.int64]
np_float_types = [np.float32, np.float64]
torch_bool_types = [torch.bool]
torch_int_types = [torch.int8, torch.int16, torch.int32, torch.int64]
torch_float_types = [torch.float32, torch.float64]


def array_to_tensor(
    array: Array, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> Tensor:
    """Converts np.ndarray and torch.Tensor to torch.Tensor with dtype and on device.
    When dtype is None, unsafe casts all float-type arrays to torch.float32 and all int-type arrays to torch.int64
    """
    input_dtype = array.dtype
    if isinstance(input_dtype, np.dtype):
        if dtype is None:
            if input_dtype in np_float_types:
                dtype = torch.float32
            elif input_dtype in np_int_types:
                dtype = torch.int64
            elif input_dtype in np_bool_types:
                dtype = torch.bool
            else:
                raise TypeError(
                    f"{input_dtype} was not a supported numpy int, float, or bool."
                )
        return torch.from_numpy(array).to(dtype=dtype, device=device)
    elif isinstance(input_dtype, torch.dtype):
        if dtype is None:
            if input_dtype in torch_float_types:
                dtype = torch.float32
            elif input_dtype in torch_int_types:
                dtype = torch.int64
            elif input_dtype in torch_bool_types:
                dtype = torch.bool
            else:
                raise TypeError(
                    f"{input_dtype} was not a supported torch int, float, or bool."
                )
        return array.to(dtype=dtype, device=device)
    else:
        raise TypeError(
            f"{input_dtype} was not recognized as a supported numpy.dtype or torch.dtype."
        )


def tobytes(x: Array):
    if isinstance(x, np.ndarray):
        return x.tobytes()
    elif isinstance(x, Tensor):
        return x.numpy().tobytes()
    else:
        raise TypeError(f"{type(x)} does not support tobytes.")


def depth(seq: Sequence):
    if seq and isinstance(seq, Sequence):
        return 1 + max(depth(item) for item in seq)
    else:
        return 0


def process_combinations(comb: Combinations):
    d = depth(comb)
    if d == 0:
        return [[comb]]
    elif d == 1:
        return [[i] for i in comb]
    elif d == 2:
        return comb
    else:
        raise ValueError(f"{comb} is not understood to be of type Combinations.")


def is_empty(directory: PathType):
    directory = Path(directory)
    if next(directory.iterdir(), None) is None:
        return True
    else:
        return False


if __name__ == "__main__":
    pass
