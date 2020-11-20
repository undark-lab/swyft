# pylint: disable=no-member
import torch
import torch.nn as nn
from numpy import ndarray

from .utils import combine_z, get_z, array_to_tensor
from .types import Array, Combinations, Device, Optional

# NOTE: z combinations (with pdim > 1) should not be generated here, but just
# fed it. They can be generated externally.


def eval_net(net, x0, z, n_batch=64):
    """Evaluates network.

    Args:
        net (nn.Module): trained ratio estimation net.
        x0 (torch.tensor): data.
        z : (nsamples, pnum, pdim)
        n_batch (int): minibatch size.

    Returns:
        net output: (nsamples, pnum)
    """
    nsamples = len(z)

    out = []
    for i in range(nsamples // n_batch + 1):
        zbatch = z[i * n_batch : (i + 1) * n_batch]
        out += net(x0.unsqueeze(0), zbatch).detach().cpu()

    return torch.stack(out)


def get_ratios(
    x0: Array,
    net: nn.Module,
    points: torch.utils.data.Dataset,
    combinations: Optional[Combinations] = None,
    device: Device = "cpu",
    max_n_points: int = 1000,
) -> ndarray:
    """From parameter z estimate the corresponding likelihood ratios.

    Args:
        x0 (Array): true observation
        net (nn.Module): likelihood ratio estimator
        points: yields parameters to train on
        combinations: parameter combinations
        device
        max_n_points: number of points to evalute ratios on
    """
    x0 = array_to_tensor(x0, device=device)
    z = get_z(points)[:max_n_points]
    z = torch.stack(z).to(device)
    z = torch.stack([combine_z(zs, combinations) for zs in z])
    ratios = eval_net(net, x0, z)
    return z.cpu().numpy(), ratios.cpu().numpy()


if __name__ == "__main__":
    pass
