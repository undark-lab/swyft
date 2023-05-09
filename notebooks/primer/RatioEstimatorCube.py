import numpy as np
import torch
#from icecream import ic
import swyft

# TODO: This could be significantly nicer
class ListToTensor:
    def __init__(self, shape, ranges, hard = True):
        self.shape = shape
        self.ranges = torch.tensor(ranges).float()
        self.hard = hard
        
    @staticmethod
    def _onehotnd(p: torch.tensor, ranges: torch.Size, weights = None):
        ndim = p.shape[-1]
        onehot = torch.zeros(p.shape[:-2] + ranges, device=p.device)
        onehot_flat = onehot.reshape((-1,) + onehot.shape[-ndim:])
        p_flat = p.reshape((-1,) + p.shape[-2:])

        if weights is not None:
            weights_flat = weights.reshape((-1,) + (p.shape[-2],))
        else:
            weights_flat = torch.ones_like(p_flat.prod(-1))
        index = p_flat.long()

        onehot_flat.index_put_(((torch.arange(onehot_flat.shape[0], device=p.device)[:, None],)
                                + tuple(index.permute(-1, 0, 1))),
                               weights_flat,
                               accumulate=True)
        return onehot
        
    def __call__(self, v, weights = None):
        """We assume all parameters are between 0 and 1"""
        device = v.device
        
        # Move points onto hypercube
        bias = self.ranges[:,0].to(device)
        width = (self.ranges[:,1]-self.ranges[:,0]).to(device)
        u = (v - bias)/width

        # Mask points outside of hypercube
        mask = torch.prod((u<=1)&(u>=0), -1)
        
        # Set weights to zero for masked points
        if weights is None:
            weights = mask.float()
        else:
            weights *= mask
        
        # Move masked points to a safe space
        u = u * mask.unsqueeze(-1)
        
        # Move onto image
        factor = torch.tensor(self.shape).float()
        w = u*factor.to(device)*0.9999999
        img = self._onehotnd(w, torch.Size(self.shape), weights = weights)
        
        return img
    
    
def points_to_image(v, w, shapes, ranges):
    ltl = ListToTensor(shapes, ranges)
    return ltl(v, w)
    
    
def var2fix(var, Nmax, fill = -np.inf):
    """Generates fixed-length array by padding with fill values."""
    N = len(var)
    assert N <= Nmax, "Nmax < N in var2fix"
    fix = torch.empty((Nmax, *var.shape[1:]), device = var.device, dtype = var.dtype)
    fix[:N] = var
    fix[N:] = fill
    return fix


def fix2var(fix, fill = -np.inf):
    """Generates var-length array by removing rows with fill values."""
    mask = fix[:,0] != fill
    return fix[mask]


def detect_peaks(x):
    """Detect peaks in map (batched)."""
    I, J = x.shape[-2:]
    m = torch.ones_like(x)
    m1 = torch.ones_like(x)
    m0 = torch.zeros_like(x)
    xp = torch.nn.functional.pad(x, (1, 1, 1, 1))
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            if i != j:
                m *= torch.where(x > xp[..., i:I+i, j:J+j], m1, m0)
    return m


class RatioEstimatorCube(torch.nn.Module):
    def __init__(self, shapes, ranges, cumsum=False, sum_image = False):
        super().__init__()
        self.shapes = shapes
        self.ranges = ranges
        self.cumsum = cumsum
        self.sum_image = sum_image
        
    def forward(self, x, z):
        """
        Args:
            x: cube
            z: list
        """
        
        counts = points_to_image(z, None, self.shapes, self.ranges)
        if self.cumsum:
            counts = counts + counts.sum(dim=-1, keepdims = True) - torch.cumsum(counts, dim=-1) # Count sources >= threshold
        mask = (counts>0).long()
#        print(x.shape, mask.shape)
        r, mask = swyft.equalize_tensors(x, mask)
        r_masked = r*mask
        if self.sum_image:
            r_masked = r_masked.sum((-3, -2))
        ratios = swyft.LogRatioSamples(params = counts, logratios = r_masked, parnames=np.array(['pdcc']))
        return ratios

    
def linear_rescale(v, v_ranges, u_ranges):
    """Rescales a tensor in its last dimension from v_ranges to u_ranges
    
    Args:
        v: (..., N)
        v_ranges: (N, 2)
        u_ranges: (N, 2)
    
    Returns:
        u: (..., N)
    """
    device = v.device
    v_ranges = torch.tensor(v_ranges)
    u_ranges = torch.tensor(u_ranges)

    # Move points onto hypercube
    v_bias = v_ranges[:,0].to(device)
    v_width = (v_ranges[:,1]-v_ranges[:,0]).to(device)
    
    # Move points onto hypercube
    u_bias = u_ranges[:,0].to(device)
    u_width = (u_ranges[:,1]-u_ranges[:,0]).to(device)

    t = (v - v_bias)/v_width
    u = (t*u_width+u_bias)
    return u