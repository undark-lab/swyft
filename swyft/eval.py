# pylint: disable=no-member
import torch
import torch.nn as nn
from numpy import ndarray

from .utils import combine_z, get_z, array_to_tensor
from .types import Array, Combinations, Device, Optional


def eval_net(
    x0: Array, net: nn.Module, z: Array, batch_size: int, device: Device = None
) -> ndarray:
    """Evaluate estimated likelihood ratios with a trained network.

    Args:
        x0 (Array): true observation
        net (nn.Module): trained ratio estimation network
        z (Array): shape (nsamples, pnum, pdim)
        batch_size (int): evaluation minibatch size
        device

    Returns:
        estimated likelihood ratio (ndarray): shape (nsamples, pnum)
    """
    net_was_training = net.training
    net.eval()

    nsamples = len(z)
    z = array_to_tensor(z, device=device)
    x0 = array_to_tensor(x0, device=device).unsqueeze(0)

    out = []
    for i in range(nsamples // batch_size + 1):
        zbatch = z[i * batch_size : (i + 1) * batch_size]
        out += net(x0, zbatch).detach().cpu()

    if net_was_training:
        net.train()
    return torch.stack(out).numpy()


def get_ratios(
    x0: Array,
    net: nn.Module,
    points: torch.utils.data.Dataset,
    combinations: Optional[Combinations] = None,
    batch_size: int = 64,
    device: Device = None,
    max_n_points: int = 1000,
) -> ndarray:
    """From parameter z estimate the corresponding likelihood ratios.

    Args:
        x0 (Array): true observation
        net (nn.Module): likelihood ratio estimator
        points: yields parameters to train on
        combinations: parameter combinations
        device
        batch_size (int): evaluation minibatch size
        max_n_points: number of points to evalute ratios on
    """
    x0 = array_to_tensor(x0, device=device)
    z = get_z(points)[:max_n_points]
    z = torch.stack(z)
    z = torch.stack([combine_z(zs, combinations) for zs in z])
    ratios = eval_net(x0, net, z, batch_size, device=device)
    return z.cpu().numpy(), ratios


if __name__ == "__main__":
    pass
