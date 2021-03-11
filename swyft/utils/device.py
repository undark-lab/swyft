from typing import Optional
from warnings import warn

import torch

from swyft.types import Device


def dict_to_device(d, device, non_blocking=False):
    return {k: v.to(device, non_blocking=non_blocking) for k, v in d.items()}


def set_device(gpu: bool = False) -> torch.device:
    """Select device, defaults to cpu."""
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
    elif gpu and not torch.cuda.is_available():
        warn("Although the gpu flag was true, the gpu is not avaliable.")
        device = torch.device("cpu")
        # torch.set_default_tensor_type("torch.FloatTensor")
    else:
        device = torch.device("cpu")
        # torch.set_default_tensor_type("torch.FloatTensor")
    return device


def get_device_if_not_none(device: Optional[Device], tensor: torch.Tensor) -> Device:
    """Returns device if not None, else returns tensor.device."""
    return tensor.device if device is None else device
