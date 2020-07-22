import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import defaultdict
import math
#from sklearn.neighbors import BallTree


#######################
# Convenience functions
#######################

def sortbyfirst(x, y):
    """Sort two lists by values of first list."""
    i = np.argsort(x)
    return x[i], y[i]

def subsample(n_sub, z, replace = False):
    """Subsample lists."""
    if n_sub is None:
        return z
    if n_sub >= len(z) and not replace:
        raise ValueError("Number of sub-samples without replacement larger than sample size")
    indices = np.random.choice(len(z), size = n_sub, replace = replace)
    z_sub = [z[i] for i in indices]
    return z_sub

def combine_z(z, combinations):
    """Generate parameter combinations."""
    if combinations is None:
        return z.unsqueeze(-1)
    else:
        return torch.stack([z[c] for c in combinations])


#########################
# Generate sample batches
#########################

def sample_z(n_draws, n_dim):
    """Return uniform samples from the hyper cube.

    Args:
        n_draws (int): Number of samples
        n_dim (int): Dimension of hypercube

    Returns:
        list: A list of length n_draws, with (n_dim,) array elements
    """
    return [np.random.rand(n_dim) for i in range(n_draws)]

def sample_x(model, zlist):
    """Generates x ~ model(z).
    
    Args:
        model (fn): Foreward model, returns samples x~p(x|z).
        zlist (list): List of model parameters z.

    Returns:
        list: List of dictionaries with 'x' and 'z' pairs.
    """
    xz = []
    n_samples = len(zlist)
    for i in tqdm(range(n_samples)):
        x = model(zlist[i])
        xz.append(dict(x=x, z=zlist[i]))
    return xz

def get_z(xz):
    """Extract z from batch of samples."""
    return [xz[i]['z'] for i in range(len(xz))]

def get_x(xz):
    """Extract x from batch of samples."""
    return [xz[i]['x'] for i in range(len(xz))]


##########
# Training
##########

def loss_fn(network, xz, combinations = None, n_batch = 32):
    """Evaluate binary-cross-entropy loss function.

    Args:
        network (nn.Module): Network taking minibatch of samples and returing ratio estimator.
        xz (list): Batch of samples to train on.
        combinations (list): Optional, determines posteriors that are generated.
            examples:
                [[0,1], [3,4]]: p(z_0,z_1) and p(z_3,z_4) are generated
                    initialize network with zdim = 2, pdim = 2
                [[0,1,5,2]]: p(z_0,z_1,z_5,z_2) is generated
                    initialize network with zdim = 1, pdim = 4
        n_batch (int): Mini-batch size.

    Returns:
        torch.tensor: Training loss
    """
    # generate minibatch
    xz = subsample(2*n_batch, xz, replace = True)
    x = get_x(xz)
    z = get_z(xz)

    # bring x into shape
    # (n_batch*2, data-shape)  - repeat twice each sample of x - there are 2*n_batch samples
    # repetition pattern in first dimension is: [a, a, b, b, c, c, d, d, ...]
    x = torch.stack(x)
    x = torch.repeat_interleave(x, 2, dim = 0)

    # bring z into shape
    # (n_batch*4, param-shape)  - repeat twice each sample of z - there are 2*n_batch samples
    # repetition is twisted in first dimension: [a, b, a, b, c, d, c, d, ...]
    z = torch.stack([combine_z(zs, combinations) for zs in z])
    zdim = len(z[0])
    z = z.view(n_batch, -1, *z.shape[-1:])
    z = torch.repeat_interleave(z, 2, dim = 0)
    z = z.view(n_batch*4, -1, *z.shape[-1:])

    # call network
    lnL = network(x, z)
    lnL = lnL.view(n_batch, 4, zdim)

    # Evaluate cross-entropy loss
    # loss = 
    # -ln( exp(lnL(x_a, z_a))/(1+exp(lnL(x_a, z_a))) )
    # -ln( exp(lnL(x_b, z_b))/(1+exp(lnL(x_b, z_b))) )
    # -ln( 1/(1+exp(lnL(x_a, z_b))) )
    # -ln( 1/(1+exp(lnL(x_b, z_a))) )
    loss  = -torch.nn.functional.logsigmoid( lnL[:,0])
    loss += -torch.nn.functional.logsigmoid(-lnL[:,1])
    loss += -torch.nn.functional.logsigmoid(-lnL[:,2])
    loss += -torch.nn.functional.logsigmoid( lnL[:,3])
    loss = loss.sum() / n_batch

    return loss

def train(network, xz, n_train = 1000, lr = 1e-3, n_batch = 32, combinations = None):
    """Network training loop.

    Args:
        network (nn.Module): Network for ratio estimation.
        xz (list): Batch of samples
        n_train (int): Training steps
        lr (float): Learning rate
        n_batch (int): Minibatch size
        combinations (list): List of parameter combinations

    Returns:
        list: List of training losses
    """
    optimizer = torch.optim.Adam(network.parameters(), lr = lr, weight_decay = 0.0000)
    losses = []
    for i in tqdm(range(n_train)):
        optimizer.zero_grad()
        loss = loss_fn(network, xz, combinations = combinations, n_batch = n_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy().item())
    return losses


######################
# Posterior estimation
######################

def estimate_lnL(network, x0, z, normalize = True, combinations = None, n_batch = 64):
    """Return current estimate of normalized marginal 1-dim lnL.

    Args:
        network (nn.Module): trained ratio estimation network
        x0 (torch.tensor): data
        z (list): list of parameter points to evaluate
        normalize (bool): set max(lnL) = 0
        combinations (list): Optional, parameter combinations
        n_batch (int): minibatch size

    Returns:
        list: List of dictionaries with component z and lnL
    """
    x0 = x0.unsqueeze(0)
    zdim = len(z[0]) if combinations is None else len(combinations)
    n_samples = len(z)

    lnL_out = []
    z_out = []
    for i in tqdm(range(n_samples//n_batch+1), desc = 'estimating lnL'):
        zbatch = z[i*n_batch:(i+1)*n_batch]
        zcomb = torch.stack([combine_z(zn, combinations) for zn in zbatch])
        tmp = network(x0, zcomb)
        lnL_out += list(tmp.detach().cpu().numpy())
        z_out += list(zcomb.cpu().numpy())

    out = []
    for i in range(zdim):
        z_i = np.array([z_out[j][i] for j in range(n_samples)])
        lnL_i= np.array([lnL_out[j][i] for j in range(n_samples)])
        if normalize:
            lnL_i -= lnL_i.max()
        out.append(dict(z=z_i, lnL=lnL_i))
    return out


##########
# Networks
##########

# From: https://github.com/pytorch/pytorch/issues/36591
class LinearWithChannel(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()

        #initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, output_size, input_size))
        self.b = torch.nn.Parameter(torch.zeros(channel_size, output_size))

        #change weights to kaiming
        self.reset_parameters(self.w, self.b)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.matmul(self.w, x).squeeze(-1) + self.b
    
def combine(y, z):
    """Combines data vectors y and parameter vectors z.
    
    z : (..., zdim, pdim)
    y : (..., ydim)
    
    returns: (..., zdim, ydim + pdim)
    
    """
    y = y.unsqueeze(-2) # (..., 1, ydim)
    y = y.expand(*z.shape[:-1], *y.shape[-1:]) # (..., zdim, ydim)
    return torch.cat([y, z], -1)

class DenseLegs(nn.Module):
    def __init__(self, ydim, zdim, pdim = 1, p = 0.0, NH = 256):
        super().__init__()
        self.fc1 = LinearWithChannel(ydim+pdim, NH, zdim)
        self.fc2 = LinearWithChannel(NH, NH, zdim)
        self.fc3 = LinearWithChannel(NH, NH, zdim)
        self.fc4 = LinearWithChannel(NH, 1, zdim)
        self.drop = nn.Dropout(p = p)

    def forward(self, y, z):
        x = combine(y, z)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x).squeeze(-1)
        return x

def get_norms(xz):
    x = get_x(xz)
    z = get_z(xz)
    x_mean = sum(x)/len(x)
    z_mean = sum(z)/len(z)
    x_var = sum([(x[i]-x_mean)**2 for i in range(len(x))])/len(x)
    z_var = sum([(z[i]-z_mean)**2 for i in range(len(z))])/len(z)
    return x_mean, x_var**0.5, z_mean, z_var**0.5

class Network(nn.Module):
    def __init__(self, xdim, zdim, pdim = 1, xz_init = None, head = None, p = 0.):
        """Base network combining z-independent head and parallel tail.

        :param xdim: Number of data dimensions going into DenseLeg network
        :param zdim: Number of latent space variables
        :param xz_init: xz Samples used for normalization
        :param head: Head network, z-independent
        :type head: `torch.nn.Module`, optional

        The forward method of the `head` network takes data `x` as input, and
        returns intermediate state `y`.
        """
        super().__init__()
        self.head = head
        self.legs = DenseLegs(xdim, zdim, pdim = pdim, p = p)
        
        if xz_init is not None:
            x_mean, x_std, z_mean, z_std = get_norms(xz_init)
        else:
            x_mean, x_std, z_mean, z_std = 0., 1., 0., 1.
        self.x_mean = torch.nn.Parameter(torch.tensor(x_mean).float())
        self.z_mean = torch.nn.Parameter(torch.tensor(z_mean).float())
        self.x_std = torch.nn.Parameter(torch.tensor(x_std).float())
        self.z_std = torch.nn.Parameter(torch.tensor(z_std).float())
    
    def forward(self, x, z):
        #TODO : Bring normalization back
        #x = (x-self.x_mean)/self.x_std
        #z = (z-self.z_mean)/self.z_std

        if self.head is not None:
            y = self.head(x)
        else:
            y = x  # Take data as features

        out = self.legs(y, z)
        return out

def iter_sample_z(n_draws, n_dim, net, x0, device = 'cpu', verbosity = False, threshold = 1e-6):
    """Generate parameter samples z~p_c(z) from constrained prior.
    
    Arguments
    ---------
    n_draws: Number of draws
    n_dim: Number of dimensions of z
    net: Trained density network
    x0: Reference data
    
    Returns
    -------
    z: list of n_dim samples with length n_draws
    """
    done = False
    zout = defaultdict(lambda: [])
    counter = np.zeros(n_dim)
    frac = np.ones(n_dim)
    while not done:
        z = sample_z(n_draws, n_dim)
        zlnL = estimate_lnL(net, x0, z, sort = False, device = device)
        for i in range(n_dim):
            mask = zlnL[i]['lnL'] > np.log(threshold)
            frac[i] = sum(mask)/len(mask)
            zout[i].append(zlnL[i]['z'][mask])
            counter[i] += mask.sum()
        done = min(counter) >= n_draws
    if verbosity:
        print("Constrained posterior volume:", frac.prod())
    out = list(np.array([np.concatenate(zout[i])[:n_draws] for i in range(n_dim)]).T[0])
    return out








#############
# OLD OLD OLD
#############

def get_posteriors(network, x0, z, error = False, n_sub = None):
    """Get posteriors, potentially with MC dropout uncertainties.
    """
    zsub = subsample(n_sub, z)
    if not error:
        network.eval()
        z_lnL = estimate_lnL(network, x0, zsub, normalize = True)
        return z_lnL
    else:
        network.train()
        z_lnL_list = []
        for i in tqdm(range(100), desc="Estimating std"):
            z_lnL = estimate_lnL(network, x0, zsub, normalize = False)
            z_lnL_list.append(z_lnL)
        std_list = []
        for j in range(len(z_lnL_list[0])):
            tmp = [z_lnL_list[i][j]['lnL'] for i in range(len(zsub))]
            mean = sum(tmp)/len(zsub)
            tmp = [(z_lnL_list[i][j]['lnL']-mean)**2 for i in range(len(zsub))]
            var = sum(tmp)/len(zsub)
            std = var**0.5
            std_list.append(std)
        return std_list

def resample_z(n, z_seeds, epsilon = None):
    z_samples = []
    n_dim = len(z_seeds)
    for i in range(n_dim):
        z = z_seeds[i].reshape(-1, 1)
        tree = BallTree(z)
        # Estimate epsilon as  epsilon = 4 * (average nn-distance)
        if epsilon is None:
          nn_dist = tree.query(z, 2)[0][:,1]
          epsilon = nn_dist.mean() * 4.
        z_new = []
        counter = 0
        while counter < n:
            z_proposal = torch.rand(n, 1).to(z_seeds[0].device);
            nn_dist = tree.query(z_proposal, 1)[0][:,0]
            mask = nn_dist <= epsilon
            z_new.append(z_proposal[mask])
            counter += mask.sum()
        z_new = torch.cat(z_new)[:n]
        z_samples.append(z_new)
    z = [torch.cat([z_samples[i][j] for i in range(n_dim)]) for j in range(n)]
    return z

def init_xz(model, n_sims, n_dim):
    z = sample_z(n_sims, n_dim)
    xz = sample_x(model, z)
    return xz

def update_xz(xz, network, x0, model, n_sims, lnL_th = -6, n_sub = 1000, append = True):
    # Generate training points
    z_sub = subsample(n_sub, get_z(xz))
    z_lnL = estimate_lnL(network, x0, z_sub)
    z_seeds = get_seeds(z_lnL, lnL_th = lnL_th)
    z_new = resample_z(n_sims, z_seeds)

    # Generate training data
    xz_new = sample_x(model, z_new)

    # Append or not append
    if append:
        xz += xz_new
    else:
        xz = xz_new
    return xz

