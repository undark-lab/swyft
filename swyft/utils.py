# pylint: disable=no-member, not-callable
from warnings import warn

import torch


def comb2d(indices):
    output = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            output.append([indices[i], indices[j]])
    return output


def combine_z(z, combinations):
    """Generate parameter combinations in last dimension.
    Requires: z.ndim == 1.
    output.shape == (n_posteriors, parameter shape)
    """
    if combinations is None:
        return z.unsqueeze(-1)
    else:
        return torch.stack([z[c] for c in combinations])


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


if __name__ == "__main__":
    pass
