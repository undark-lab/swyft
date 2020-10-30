# pylint: disable=no-member
import torch

from .utils import combine_z, get_z

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


def get_ratios(x0, net, dataset, combinations=None, device="cpu", Nmax=1000):
    x0 = x0.to(device)
    z = get_z(dataset)[:Nmax]
    z = torch.stack(z).to(device)
    z = torch.stack([combine_z(zs, combinations) for zs in z])
    ratios = eval_net(net, x0, z)
    return z.cpu(), ratios.cpu()


if __name__ == "__main__":
    pass
