import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.neighbors import BallTree

################
# Core functions
################

def sample_z(n_draws, n_dim):
    return [torch.rand(n_dim) for i in range(n_draws)]

def sample_x(model, z):
    xz = []
    n_samples = len(z)
    for i in tqdm(range(n_samples)):
        x = model(z[i])
        x = torch.tensor(x, dtype=torch.float32)
        xz.append(dict(x=x, z=z[i]))
    return xz

def train(network, xz, n_steps = 1000, lr = 1e-3, n_particles = 4):
    # 1. Randomly select n_particles from train_data
    # 2. Calculate associated loss by permuting
    # 3. Repeat n_step times

    def loss_fn(network, xz, n_particles = 3):
        indices = np.random.choice(len(xz), size = n_particles, replace = False)
        x = [xz[i]['x'] for i in indices]
        z = [xz[i]['z'] for i in indices]
        loss = torch.tensor(0.)
        for i in range(n_particles):
            f = [network(x[i], z[j]).unsqueeze(0) for j in range(n_particles)]
            f = torch.cat(f, 0)
            g = torch.log_softmax(f, 0)
            particle_loss = -g[i]
            loss += particle_loss.sum()
        return loss

    optimizer = torch.optim.Adam(network.parameters(), lr = lr)
    losses = []

    for i in tqdm(range(n_steps)):
        optimizer.zero_grad()

        loss = loss_fn(network, xz, n_particles = n_particles)
        losses.append(loss.detach().numpy().item())
        loss.backward()
        optimizer.step()

    return losses

def get_z(xz):
    return [xz[i]['z'] for i in range(len(xz))]

def get_x(xz):
    return [xz[i]['x'] for i in range(len(xz))]

def estimate_lnL(network, x0, z, L_th = 1e-3, n_train = 10, epsilon = 1e-2, n_sub = 1000):
    """Return current estimate of normalized marginal 1-dim lnL.  Returns (n_train, n_dim)."""
    if n_sub > 0:
        z = subsample(n_sub, z)
    x0 = torch.tensor(x0, dtype=torch.float32)
    n_dim = z[0].shape[0]
    lnL = [network(x0, z[i]).detach() for i in range(len(z))]
    out = []
    for i in range(n_dim):
        tmp = [[z[j][i], lnL[j][i]] for j in range(len(z))]
        tmp = sorted(tmp, key = lambda pair: pair[0])
        z_i = np.array([y[0] for y in tmp])
        lnL_i = np.array([y[1] for y in tmp])
        lnL_i -= lnL_i.max()
        out.append(dict(z=z_i, lnL=lnL_i))
    return out

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
            z_proposal = torch.rand(n, 1);
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


###################
# Training networks
###################

class MLP(nn.Module):
    def __init__(self, n_dim = None, n_hidden = None, xz_init = None):
        super().__init__()
        self.n_dim = n_dim
        self.fc1 = nn.ModuleList([nn.Linear(n_dim+1, n_hidden) for i in range(n_dim)])
        self.fc2 = nn.ModuleList([nn.Linear(n_hidden, 1) for i in range(n_dim)])

        if xz_init is not None:
            self.normalize = True
            tmp = self._get_norms(xz_init)
            self.x_mean, self.x_std, self.z_mean, self.z_std = tmp
        else:
            self.normalize = False

    @staticmethod
    def _get_norms(xz):
        x = get_x(xz)
        z = get_z(xz)
        x_mean = sum(x)/len(x)
        z_mean = sum(z)/len(z)
        x_var = sum([(x[i]-x_mean)**2 for i in range(len(x))])/len(x)
        z_var = sum([(z[i]-z_mean)**2 for i in range(len(z))])/len(z)
        return x_mean, x_var**0.5, z_mean, z_var**0.5

    def _normalized(self, x, z):
        return (x-self.x_mean)/self.x_std, (z-self.z_mean)/self.z_std

    def forward(self, x, z):
        if self.normalize:
            x, z = self._normalized(x, z)

        f_list = []
        for i in range(self.n_dim):
            y = x
            y = torch.cat([y, z[i].unsqueeze(0)], 0)
            y = torch.relu(self.fc1[i](y))
            f = self.fc2[i](y)
            f_list.append(f)
        f_list = torch.cat(f_list, 0)
        return f_list
