import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.neighbors import BallTree
from collections import defaultdict
import math

################
# Core functions
###############

def sample_z(n_draws, n_dim):
    """Return uniform samples from the hyper cube.

    Args:
        n_draws (int): Number of samples
        n_dim (int): Dimension of hypercube

    Returns:
        list: A list of length n_draws, with (n_dim,) array elements
    """
    return [np.random.rand(n_dim) for i in range(n_draws)]

def sortbyfirst(x, y):
    i = np.argsort(x)
    return x[i], y[i]

def sample_x(model, z):
    """Augments parameter points with simulated data.
    
    :param model: Function x = model(z[0]) etc.
    :param z: List of parameter points.
    :rtype: List of {'x':x, 'z':z) dicts.
    """
    xz = []
    n_samples = len(z)
    for i in tqdm(range(n_samples)):
        x = model(z[i])
        xz.append(dict(x=x, z=z[i]))
    return xz

def get_z(xz):
    return [xz[i]['z'] for i in range(len(xz))]

def get_x(xz):
    return [xz[i]['x'] for i in range(len(xz))]

def loss_fn_batched(network, xz, combinations = None, device = 'cpu', n_batch = 1):
    xz = subsample(2*n_batch, xz, replace = True)
    x = get_x(xz)
    z = get_z(xz)

    # bring into shape
    x = torch.tensor([a for a in x]).float().to(device)
    x = torch.repeat_interleave(x, 2, dim = 0)
    # (n_batch*4, data-shape)  - repeat twice each sample of x - there are 2*n_batch samples
    # repetition pattern in first dimension is: [a, a, b, b, c, c, d, d, ...]

    z = torch.tensor([combine2(zs, combinations) for zs in z]).float().to(device)
    zdim = len(z[0])
    z = z.view(n_batch, -1, *z.shape[-1:])
    z = torch.repeat_interleave(z, 2, dim = 0)
    z = z.view(n_batch*4, -1, *z.shape[-1:])
    # (n_batch*4, param-shape)  - repeat twice each sample of z - there are 2*n_batch samples
    # repetition is twisted in first dimension: [a, b, a, b, c, d, c, d, ...]

    lnL = network(x, z)
    lnL = lnL.view(n_batch, 4, zdim)

    loss  = -torch.nn.functional.logsigmoid( lnL[:,0])
    loss += -torch.nn.functional.logsigmoid(-lnL[:,1])
    loss += -torch.nn.functional.logsigmoid(-lnL[:,2])
    loss += -torch.nn.functional.logsigmoid( lnL[:,3])

    loss = loss.sum() / n_batch
    return loss

#def loss_fn(network, xz, combinations = None, device = 'cpu'):
#    xz = subsample(2, xz)
#    x = get_x(xz)
#    z = get_z(xz)
#
#    x = [torch.tensor(a).float().to(device) for a in x]
#
#    # z has to be list of (zdim, pdim) arrays
#    if combinations is None:
#        z = [torch.tensor(a).float().to(device).unsqueeze(-1) for a in z]
#    else:
#        z = [torch.stack([torch.tensor(a).float().to(device)[c] for c in combinations]) for a in z]
#
#    lnL_r = [
#            network(x[0], z[0]).unsqueeze(0),
#            network(x[0], z[1]).unsqueeze(0),
#            network(x[1], z[1]).unsqueeze(0),
#            network(x[1], z[0]).unsqueeze(0)]
#
#    loss  = -torch.nn.functional.logsigmoid( lnL_r[0])
#    loss += -torch.nn.functional.logsigmoid(-lnL_r[1])
#    loss += -torch.nn.functional.logsigmoid( lnL_r[2])
#    loss += -torch.nn.functional.logsigmoid(-lnL_r[3])
#    return loss.sum()


#    def loss_fn(network, xz, n_particles = 3):
#        if n_particles == 1:
#            return loss_fn2(network, xz)
#
#        xz = subsample(n_particles, xz)
#        x = get_x(xz)
#        z = get_z(xz)
#        #indices = np.random.choice(len(xz), size = n_particles, replace = False)
#        #x = [xz[i]['x'] for i in indices]
#        #z = [xz[i]['z'] for i in indices]
#
#        x = [torch.tensor(a).float().to(device) for a in x]
#        z = [torch.tensor(a).float().to(device) for a in z]
#
#        loss = torch.tensor(0.).to(x[0].device)
#        for i in range(n_particles):
#            f = [network(x[i], z[j]).unsqueeze(0) for j in range(n_particles)]
#            f = torch.cat(f, 0)
#            g = torch.log_softmax(f, 0)
#            particle_loss = -g[i]
#            loss += particle_loss.sum()
#        return loss

def train(network, xz, n_train = 1000, lr = 1e-3, device = 'cpu', n_batch = 3, combinations = None):
    """Train a network.
    """
    optimizer = torch.optim.Adam(network.parameters(), lr = lr, weight_decay = 0.0000)
    losses = []
    for i in tqdm(range(n_train)):
        optimizer.zero_grad()
        loss = loss_fn_batched(network, xz, combinations = combinations, device = device, n_batch = n_batch)
        #loss = 0.
        #for j in range(n_batch):
        #    loss += loss_fn(network, xz, combinations = combinations, device = device)
        #loss /= n_batch
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy().item())
    return losses

def combine2(z, combinations):
    if combinations is None:
        return np.stack([[z[i]] for i in range(len(z))])
    else:
        return np.stack([z[c] for c in combinations])

def estimate_lnL_batched(network, x0, z, sort = True, device = 'cpu', normalize = True, combinations = None, n_batch = 64):
    """Return current estimate of normalized marginal 1-dim lnL.  List of zdim dictionaries."""
    x0 = torch.tensor(x0).float().to(device)
    zdim = len(z[0]) if combinations is None else len(combinations)
    n_samples = len(z)

    lnL_out = []
    z_out = []
    for i in tqdm(range(n_samples//n_batch+1), desc = 'estimating lnL'):
        zbatch = z[i*n_batch:(i+1)*n_batch]
        zcomb_list = [combine2(zn, combinations) for zn in zbatch]
        zcomb = torch.tensor(zcomb_list).float().to(device)
        tmp = network(x0, zcomb)
        lnL_list = list(tmp.detach().cpu().numpy())
        lnL_out += lnL_list
        z_out += zcomb_list

    out = []
    for i in range(zdim):
        z_i = np.array([z_out[j][i] for j in range(n_samples)])
        lnL_i= np.array([lnL_out[j][i] for j in range(n_samples)])
        if normalize:
            lnL_i -= lnL_i.max()
        out.append(dict(z=z_i, lnL=lnL_i))
    return out

def get_posteriors(network, x0, z, device = 'cpu', error = False, n_sub = None):
    zsub = subsample(n_sub, z)
    if not error:
        network.eval()
        z_lnL = estimate_lnL_batched(network, x0, zsub, device = device, normalize = True)
        return z_lnL
    else:
        network.train()
        z_lnL_list = []
        for i in tqdm(range(100), desc="Estimating std"):
            z_lnL = estimate_lnL_batched(network, x0, zsub, device = device, normalize = False)
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

#def estimate_lnL(network, x0, z, n_sub = 0, sort = True, device = 'cpu', normalize = True, combinations = None):
#    """Return current estimate of normalized marginal 1-dim lnL.  List of n_dim dictionaries."""
#    if n_sub > 0:
#        z = subsample(n_sub, z)
#    if combinations is None:
#        n_dim = len(z[0])
#    else:
#        n_dim = len(combinations)
#    x0 = torch.tensor(x0).float().to(device)
#    lnL = [
#            network(x0, torch.tensor(combine2(zn, combinations)).float().to(device)).detach().cpu().numpy() for zn in z]
#    out = []
#    for i in range(n_dim):
#        tmp = [[combine2(z[j], combinations)[i], lnL[j][i]] for j in range(len(z))]
#        if sort:
#            tmp = sorted(tmp, key = lambda pair: pair[0])
#        z_i = np.array([y[0] for y in tmp])
#        lnL_i = np.array([y[1] for y in tmp])
#        if normalize:
#            lnL_i -= lnL_i.max()
#        out.append(dict(z=z_i, lnL=lnL_i))
#    return out

#def estimate_lnL_2d(network, x0, z, n_sub = 1000):
#    """Returns single dict(z, lnL)"""
#    if n_sub > 0:
#        z = subsample(n_sub, z)
#    lnL = np.array([network(x0, z[k]).detach() for k in range(len(z))])
#    lnL -= lnL.max()
#    return dict(z = z, lnL = lnL)
#
#def get_seeds(z_lnL, lnL_th = -6):
#    z_seeds = []
#    for i in range(len(z_lnL)):
#        z = z_lnL[i]['z']
#        lnL = z_lnL[i]['lnL']
#        mask = lnL > lnL_th
#        z_seeds.append(z[mask])
#    return z_seeds

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

def subsample(n_sub, z, replace = False):
    """Subsample lists."""
    if n_sub is None:
        return z
    if n_sub >= len(z) and not replace:
        raise ValueError("Number of sub-samples without replacement larger than sample size")
    indices = np.random.choice(len(z), size = n_sub, replace = replace)
    z_sub = [z[i] for i in indices]
    return z_sub

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

def get_norms(xz):
    x = get_x(xz)
    z = get_z(xz)
    x_mean = sum(x)/len(x)
    z_mean = sum(z)/len(z)
    x_var = sum([(x[i]-x_mean)**2 for i in range(len(x))])/len(x)
    z_var = sum([(z[i]-z_mean)**2 for i in range(len(z))])/len(z)
    return x_mean, x_var**0.5, z_mean, z_var**0.5

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

#def combine(y, z):
#    """Combines data vector y and parameter vector z.
#
#    z : (zdim) or (nbatch, zdim)
#    y : (ydim) or (nbatch, ydim) (only if nbatch provided for z)
#
#    returns: (nbatch, zdim, ydim+1)
#    """
#    y = y.unsqueeze(-2)
#    z = z.unsqueeze(-1)
#    y = y.expand(*z.shape[:-1], *y.shape[-1:])
#    return torch.cat([y, z], -1)

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

class ConvHead(nn.Module):
    def __init__(self, xdim):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 10, 3)
        self.conv2 = torch.nn.Conv1d(10, 20, 3)
        self.conv3 = torch.nn.Conv1d(20, 30, 3)
        self.pool = torch.nn.MaxPool1d(2)
        
    def forward(self, x):
        """Input (nbatch, xdim)"""
        x = x.unsqueeze(-2)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x.flatten(start_dim=-2)

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
        zlnL = estimate_lnL_batched(net, x0, z, sort = False, device = device)
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



###################
# Training networks
###################

class MLP(nn.Module):
    def __init__(self, x_dim, z_dim, n_hidden, xz_init = None):
        """Model for marginal 1-dim posteriors, p(z_1|x), p(z_2|x), ..., p(z_N|x)"""
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.fc1 = nn.ModuleList([nn.Linear(x_dim+1, n_hidden) for i in range(z_dim)])
        self.fc2 = nn.ModuleList([nn.Linear(n_hidden, 1) for i in range(z_dim)])

        if xz_init is not None:
            self.normalize = True
            tmp = get_norms(xz_init)
            self.x_mean, self.x_std, self.z_mean, self.z_std = tmp
        else:
            self.normalize = False

    def forward(self, x, z):
        if self.normalize:
            x = (x-self.x_mean)/self.x_std
            z = (z-self.z_mean)/self.z_std

        f_list = []
        for i in range(self.z_dim):
            y = x
            y = torch.cat([y, z[i].unsqueeze(0)], 0)
            y = torch.relu(self.fc1[i](y))
            f = self.fc2[i](y)
            f_list.append(f)
        f_list = torch.cat(f_list, 0)
        return f_list


class MLP_2d(nn.Module):
    def __init__(self, x_dim, n_hidden, xz_init = None):
        """Model for joined 2-dim posterior, p(z_1, z_2|x)"""
        super().__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim+2, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

        if xz_init is not None:
            self.normalize = True
            tmp = get_norms(xz_init)
            self.x_mean, self.x_std, self.z_mean, self.z_std = tmp
        else:
            self.normalize = False

    def forward(self, x, z):
        if self.normalize:
            x = (x-self.x_mean)/self.x_std
            z = (z-self.z_mean)/self.z_std

        y = torch.cat([x, z], 0)
        y = torch.relu(self.fc1(y))
        f = self.fc2(y)
        return f
