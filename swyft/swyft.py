import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from .core import *

class SWYFT:
    """SWYFT. Convenience class around functional methods."""
    def __init__(self, model, z_dim, x0, n_train = 10000, n_sims = 10000, n_hidden = 10000):
        self.model = model
        self.z_dim = z_dim
        self.x0 = x0 
        self.x_dim = x0.shape[0]
        self.n_train = n_train
        self.n_sims = n_sims
        self.n_hidden = n_hidden

        self.xz_storage = []
        self.network_storage = []
        self.losses_storage = []

#    def strip(self, xz):
#        xz0 = []
#        for i in range(len(xz)):
#            z = xz[i]['z']
#            if (1.00 > z[0] and z[0] > 0.7) or (0.00 < z[0] and z[0] < 0.3):
#                xz0.append(xz[i])
#            else:
#                if np.random.random(1) < 0.1:
#                    xz0.append(xz[i])
#        return xz0

    def run(self):
        # Generate new training data
        if len(self.xz_storage) == 0:
            xz = init_xz(self.model, n_sims = self.n_sims, n_dim = self.z_dim)
        else:
            xz_prev = self.xz_storage[-1]
            network_prev = self.network_storage[-1]
            xz = update_xz(xz_prev, network_prev, self.x0, self.model, n_sims = self.n_sims, lnL_th = -9, append = False)

        # Remove some points
        #print(len(xz))
        #xz = self.strip(xz)
        #print(len(xz))

        # Instantiate network
        network = MLP(self.x_dim, self.z_dim, self.n_hidden, xz_init = xz)

        # Perform optimization
        losses = train(network, xz, n_steps = self.n_train, lr = 1e-3, n_particles = 3)
        losses = train(network, xz, n_steps = self.n_train, lr = 1e-4, n_particles = 3)

        # Store results
        self.xz_storage.append(xz)
        self.network_storage.append(network)
        self.losses_storage.append(losses)

    def get_post1d(self, round = -1):
        network = self.network_storage[round]
        z = get_z(self.xz_storage[round])
        z_lnL = estimate_lnL(network, self.x0, z, L_th=1e-5)
        return z_lnL
