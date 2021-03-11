from typing import Dict, List

import numpy as np
import torch

from swyft.types import Array


def swyftify_params(params: Array, parameter_names: List[str]) -> Dict[str, Array]:
    """Translates a [..., dim] tensor into a dictionary with dim keys.

    Args:
        params (Array):

    Returns:
        swyft_parameters (dict):
    """
    return {k: params[..., i] for i, k in enumerate(parameter_names)}


def unswyftify_params(
    swyft_params: Dict[str, Array], parameter_names: List[str]
) -> Array:
    """Translates a dictionary with dim keys into a tensor with [..., dim] shape.

    Args:
        swyft_params (Dict[str, Array]): dictionary with parameter_names as keys, tensors as values
        parameter_names (List[str]):

    Returns:
        Array: stacked params
    """
    if isinstance(swyft_params[parameter_names[0]], torch.Tensor):
        return torch.stack([swyft_params[name] for name in parameter_names], dim=-1)
    # elif isinstance(swyft_params[parameter_names[0]], np.ndarray):
    else:
        return np.stack([swyft_params[name] for name in parameter_names], axis=-1)


def swyftify_observation(observation: torch.Tensor):
    assert observation.ndim == 1, f"ndim was {observation.ndim}, but should be 1."
    return dict(x=observation)


def unswyftify_observation(swyft_observation: dict):
    return swyft_observation["x"]
