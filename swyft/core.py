import numpy as np
from tqdm import tqdm
import torch
from sklearn.neighbors import BallTree

def sample_z(n_draws, n_dim, z_seeds = None):
    """Sample latent space parameters zimport torch.functional as F
.

    Arguments
    ---------
    n_draws : Number of draws
    n_dim : Number of dimensions
    z_seeds : Optional parameter providing a list of seed positions

    Returns
    -------
    List of z samples.
    """
    if z_seeds is None:
        # Draw from unit hypercube
        return [torch.rand(n_dim) for i in range(n_draws)]
    else:
        # TODO: Finish
        pass

def sample_x(model, z):
    xz = []
    n_samples = len(z)
    for i in tqdm(range(n_samples)):
        x = model(z[i])
        x = torch.tensor(x, dtype=torch.float32)
        xz.append(dict(x=x, z=z[i]))
    return xz

def get_normalization(xz):
    return norms

def apply_norms(xz, norms):
    return xz_norm

def unapply_norms(xz, norms):
    return xz_norm

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

def estimate_lnL(network, x0, z, L_th = 1e-3, n_train = 10, epsilon = 1e-2):
    """Return current estimate of normalized marginal 1-dim lnL.  Returns (n_train, n_dim)."""
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

def resample(z_seeds, n, epsilon = None):
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
