import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.neighbors import BallTree
from collections import defaultdict
import math

################
# Core functions
################

def sample_z(n_draws, n_dim):
    return [np.random.rand(n_dim) for i in range(n_draws)]

def sample_x(model, z):
    xz = []
    n_samples = len(z)
    for i in tqdm(range(n_samples)):
        x = model(z[i])
        xz.append(dict(x=x, z=z[i]))
    return xz

def train(network, xz, n_steps = 1000, lr = 1e-3, n_particles = 2, device = 'cpu', xz_test = None, n_batch = 1):
    # 1. Randomly select n_particles from train_data
    # 2. Calculate associated loss by permuting
    # 3. Repeat n_step times

    def loss_fn2(network, xz):
        indices = np.random.choice(len(xz), size = 2, replace = False)
        x = [xz[i]['x'] for i in indices]
        z = [xz[i]['z'] for i in indices]

        x = [torch.tensor(a).float().to(device) for a in x]
        z = [torch.tensor(a).float().to(device) for a in z]

        lnL_r = [
                network(x[0], z[0]).unsqueeze(0),
                network(x[0], z[1]).unsqueeze(0),
                network(x[1], z[1]).unsqueeze(0),
                network(x[1], z[0]).unsqueeze(0)]
        #d = [torch.exp(lnL_r[i])/(1+torch.exp(lnL_r[i])) for i in range(4)]
        #loss  = -torch.log(d[0]) - torch.log(1-d[1])
        #loss -= torch.log(d[2]) + torch.log(1-d[3])
        #return loss.sum()

        loss  = -torch.nn.functional.logsigmoid( lnL_r[0])
        loss += -torch.nn.functional.logsigmoid(-lnL_r[1])
        loss += -torch.nn.functional.logsigmoid( lnL_r[2])
        loss += -torch.nn.functional.logsigmoid(-lnL_r[3])
        return loss.sum()

    def loss_fn(network, xz, n_particles = 3):
        if n_particles == 1:
            return loss_fn2(network, xz)

        indices = np.random.choice(len(xz), size = n_particles, replace = False)
        x = [xz[i]['x'] for i in indices]
        z = [xz[i]['z'] for i in indices]

        x = [torch.tensor(a).float().to(device) for a in x]
        z = [torch.tensor(a).float().to(device) for a in z]

        loss = torch.tensor(0.).to(x[0].device)
        for i in range(n_particles):
            f = [network(x[i], z[j]).unsqueeze(0) for j in range(n_particles)]
            f = torch.cat(f, 0)
            g = torch.log_softmax(f, 0)
            particle_loss = -g[i]
            loss += particle_loss.sum()
        return loss

    optimizer = torch.optim.Adam(network.parameters(), lr = lr)
    losses = []
    losses_test = []

    for i in tqdm(range(n_steps)):
        optimizer.zero_grad()

        if xz_test is not None and i%5 == 0:
            loss = loss_fn(network, xz_test, n_particles = n_particles)
            losses_test.append(loss.detach().cpu().numpy().item())
        else:
            loss = 0.
            for j in range(n_batch):
                loss += loss_fn(network, xz, n_particles = n_particles)
            loss /= n_batch
            losses.append(loss.detach().cpu().numpy().item())
            loss.backward()
            optimizer.step()

    if xz_test is None:
        return losses
    else:
        return losses, losses_test

def get_z(xz):
    return [xz[i]['z'] for i in range(len(xz))]

def get_x(xz):
    return [xz[i]['x'] for i in range(len(xz))]

def estimate_lnL(network, x0, z, n_sub = 1000, sort = True, device = 'cpu'):
    """Return current estimate of normalized marginal 1-dim lnL.  List of n_dim dictionaries."""
    if n_sub > 0:
        z = subsample(n_sub, z)
    n_dim = z[0].shape[0]
    x0 = torch.tensor(x0).float().to(device)
    lnL = [network(x0, torch.tensor(z[i]).float().to(device)).detach().cpu().numpy() for i in range(len(z))]
    out = []
    for i in range(n_dim):
        tmp = [[z[j][i], lnL[j][i]] for j in range(len(z))]
        if sort:
            tmp = sorted(tmp, key = lambda pair: pair[0])
        z_i = np.array([y[0] for y in tmp])
        lnL_i = np.array([y[1] for y in tmp])
        lnL_i -= lnL_i.max()
        out.append(dict(z=z_i, lnL=lnL_i))
    return out

def estimate_lnL_2d(network, x0, z, n_sub = 1000):
    """Returns single dict(z, lnL)"""
    if n_sub > 0:
        z = subsample(n_sub, z)
    lnL = np.array([network(x0, z[k]).detach() for k in range(len(z))])
    lnL -= lnL.max()
    return dict(z = z, lnL = lnL)

def get_seeds(z_lnL, lnL_th = -6):
    z_seeds = []
    for i in range(len(z_lnL)):
        z = z_lnL[i]['z']
        lnL = z_lnL[i]['lnL']
        mask = lnL > lnL_th
        z_seeds.append(z[mask])
    return z_seeds

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
    if n_sub >= len(z):
        return z
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
    """Combines data vector y and parameter vector z.

    z : (zdim) or (nbatch, zdim)
    y : (ydim) or (nbatch, ydim) (only if nbatch provided for z)

    returns: (nbatch, zdim, ydim+1)
    """
    y = y.unsqueeze(-2)
    z = z.unsqueeze(-1)
    y = y.expand(*z.shape[:-1], *y.shape[-1:])
    return torch.cat([y, z], -1)

class DenseLegs(nn.Module):
    def __init__(self, ydim, zdim, p = 0.0):
        super().__init__()
        NH = 1000
        self.fc1 = LinearWithChannel(ydim+1, NH, zdim)
        self.fc2 = LinearWithChannel(NH, NH, zdim)
        self.fc3 = LinearWithChannel(NH, NH, zdim)
        self.fc4 = LinearWithChannel(NH, NH, zdim)
        self.fc5 = LinearWithChannel(NH, 1, zdim)
        self.drop = nn.Dropout(p = p)

    def forward(self, y, z):
        x = combine(y, z)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        x = torch.relu(self.fc3(x))
        x = self.drop(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x).squeeze(-1)
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
    def __init__(self, xdim, zdim, xz_init = None, head = None, p = 0.):
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
        self.legs = DenseLegs(xdim, zdim, p = p)
        
        if xz_init is not None:
            x_mean, x_std, z_mean, z_std = get_norms(xz_init)
        else:
            x_mean, x_std, z_mean, z_std = 0., 1., 0., 1.
        self.x_mean = torch.nn.Parameter(torch.tensor(x_mean).float())
        self.z_mean = torch.nn.Parameter(torch.tensor(z_mean).float())
        self.x_std = torch.nn.Parameter(torch.tensor(x_std).float())
        self.z_std = torch.nn.Parameter(torch.tensor(z_std).float())
    
    def forward(self, x, z):
        x = (x-self.x_mean)/self.x_std
        z = (z-self.z_mean)/self.z_std
        
        if self.head is not None:
            x = self.head(x)
        out = self.legs(x, z)
        return out


def iter_sample_z(n_draws, n_dim, net, x0, device = 'cpu', verbosity = False):
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
            mask = zlnL[i]['lnL'] > -13
            frac[i] = sum(mask)/len(mask)
            zout[i].append(zlnL[i]['z'][mask])
            counter[i] += mask.sum()
        done = min(counter) >= n_draws
    if verbosity:
        print("Constrained posterior volume:", frac.prod())
    return np.array([np.concatenate(zout[i])[:n_draws] for i in range(n_dim)]).T


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
