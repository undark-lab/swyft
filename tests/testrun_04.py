#!/usr/bin/env python3

import matplotlib as mpl
mpl.use("Agg")
import pylab as plt
import torch
import torch.nn as nn
import numpy as np
import swyft

n_sims = 10000
n_steps = 10000

n_hidden = 200
n_particles = 2

# Definition of trivial test model
def model(z, sigma = 0.05):
    y = ((z[0]-0.5)**2 + (z[1]-0.5)**2)**0.5  # Radius
    n = np.random.randn(2) * sigma
    return y + n

class Network(nn.Module):
    def __init__(self, x_dim, z_dim, n_hidden, xz_init = None):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.fc1 = nn.ModuleList([nn.Linear(x_dim+1, n_hidden) for i in range(z_dim)])
        self.fc2 = nn.ModuleList([nn.Linear(n_hidden+1, n_hidden) for i in range(z_dim)])
        self.fc3 = nn.ModuleList([nn.Linear(n_hidden+1, n_hidden) for i in range(z_dim)])
        self.fc4 = nn.ModuleList([nn.Linear(n_hidden+1, n_hidden) for i in range(z_dim)])
        self.fc5 = nn.ModuleList([nn.Linear(n_hidden, 1) for i in range(z_dim)])

        if xz_init is not None:
            self.normalize = True
            tmp = self._get_norms(xz_init)
            self.x_mean, self.x_std, self.z_mean, self.z_std = tmp
            print("x:", self.x_mean, self.x_std)
            print("z:", self.z_mean, self.z_std)
        else:
            self.normalize = False

    @staticmethod
    def _get_norms(xz):
        x = swyft.get_x(xz)
        z = swyft.get_z(xz)
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
        for i in range(self.z_dim):
            y = x
            y = torch.cat([y, z[i].unsqueeze(0)], 0)
            y = torch.relu(self.fc1[i](y))
            y = torch.cat([y, z[i].unsqueeze(0)], 0)
            y = torch.relu(self.fc2[i](y))
            y = torch.cat([y, z[i].unsqueeze(0)], 0)
            y = torch.relu(self.fc3[i](y))
            y = torch.cat([y, z[i].unsqueeze(0)], 0)
            y = torch.relu(self.fc4[i](y))
            f = self.fc5[i](y)
            f_list.append(f)
        f_list = torch.cat(f_list, 0)
        return f_list


# Generate test z0 and x0
z0 = np.array([0.10, 0.50])
x0 = model(z0)
x_dim = len(x0)
z_dim = len(z0)

# Initialize loss list
losses = []

# And the first run
xz1 = swyft.init_xz(model, n_sims = n_sims, n_dim = z_dim)
network1 = Network(x_dim, z_dim, n_hidden, xz_init = xz1)
losses += swyft.train(network1, xz1, n_steps = n_steps, lr = 1e-3, n_particles = n_particles)
losses += swyft.train(network1, xz1, n_steps = n_steps, lr = 1e-4, n_particles = n_particles)

# Plot results
z_lnL = swyft.estimate_lnL(network1, x0, swyft.get_z(xz1))
plt.plot(z_lnL[0]['z'], np.exp(z_lnL[0]['lnL']))
plt.axvline(0.1)
plt.axvline(0.9)
plt.savefig("figs/testrun_04.png")
