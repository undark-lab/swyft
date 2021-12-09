from typing import Dict, Hashable, Optional, Union

import numpy as np
import torch

from swyft.types import Array, Device


def dict_to_device(d, device, non_blocking=False):
    return {k: v.to(device, non_blocking=non_blocking) for k, v in d.items()}


def dict_array_to_tensor(
    d, device="cpu", non_blocking=False, indices=slice(0, None)
) -> Dict[Hashable, torch.Tensor]:
    return {
        k: array_to_tensor(v[indices]).to(device, non_blocking=non_blocking)
        for k, v in d.items()
    }


np_bool_types = [bool]
np_int_types = [np.int8, np.int16, np.int32, np.int64]
np_float_types = [np.float32, np.float64]
np_complex_types = [np.complex64, np.complex128]
torch_bool_types = [torch.bool]
torch_int_types = [torch.int8, torch.int16, torch.int32, torch.int64]
torch_float_types = [torch.float32, torch.float64]
torch_complex_types = [torch.complex64, torch.complex128]


def array_to_tensor(
    array: Array, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """Converts np.ndarray and torch.Tensor to torch.Tensor with dtype and on device.
    When dtype is None, unsafe casts all float-type arrays to torch.float32 and all int-type arrays to torch.int64
    """
    if not isinstance(array, (np.ndarray, torch.Tensor)):
        array = np.asarray(array)

    input_dtype = array.dtype
    if isinstance(input_dtype, np.dtype):
        if dtype is None:
            if input_dtype in np_float_types:
                dtype = torch.float32
            elif input_dtype in np_int_types:
                dtype = torch.int64
            elif input_dtype in np_bool_types:
                dtype = torch.bool
            elif input_dtype in np_complex_types:
                dtype = torch.complex64
            else:
                raise TypeError(
                    f"{input_dtype} was not a supported numpy int, float, bool, or complex."
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
            elif input_dtype in torch_complex_types:
                dtype = torch.complex64
            else:
                raise TypeError(
                    f"{input_dtype} was not a supported torch int, float, bool, or complex."
                )
        return array.to(dtype=dtype, device=device)
    else:
        raise TypeError(
            f"{input_dtype} was not recognized as a supported numpy.dtype or torch.dtype."
        )


def tensor_to_array(
    tensor: Array, dtype: Optional[np.dtype] = None, copy: bool = True
) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        out = np.asarray(tensor.detach().cpu().numpy(), dtype=dtype)
    else:
        out = np.asarray(tensor, dtype=dtype)

    if copy:
        return out.copy()
    else:
        return out


def tobytes(x: Array) -> Array:
    if isinstance(x, np.ndarray):
        return x.tobytes()
    elif isinstance(x, torch.Tensor):
        return x.numpy().tobytes()
    else:
        raise TypeError(f"{type(x)} does not support tobytes.")


def _all_finite(x: Array) -> Array:
    if isinstance(x, torch.Tensor):
        return torch.all(torch.isfinite(x))
    else:
        return np.all(np.isfinite(x))


def all_finite(x: Union[dict, torch.Tensor, np.ndarray, list]) -> bool:
    if isinstance(x, dict):
        return all(_all_finite(v) for v in x.values())
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        return _all_finite(x)
    elif isinstance(x, list):
        return all(_all_finite(v) for v in x)
    else:
        raise NotImplementedError("That type is not yet implemented.")
