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

n_hidden = 10000
n_particles = 2

# Definition of trivial test model
def model(z, sigma = 0.05):
    y = ((z[0]-0.5)**2 + (z[1]-0.5)**2)**0.5  # Radius
    n = np.random.randn(2) * sigma
    return y + n

# Generate test z0 and x0
z0 = np.array([0.10, 0.50])
x0 = model(z0)
x_dim = len(x0)
z_dim = len(z0)

# Initialize loss list
losses = []

# And the first run
xz1 = swyft.init_xz(model, n_sims = n_sims, n_dim = z_dim)
network1 = swyft.MLP(x_dim, z_dim, n_hidden, xz_init = xz1)
losses += swyft.train(network1, xz1, n_steps = n_steps, lr = 1e-3, n_particles = n_particles)
losses += swyft.train(network1, xz1, n_steps = n_steps, lr = 1e-4, n_particles = n_particles)

xz2 = swyft.update_xz(xz1, network1, x0, model, n_sims = n_sims, lnL_th = np.log(1e-5), append=False)
network2 = swyft.MLP(x_dim, z_dim, n_hidden, xz_init = xz2)
losses += swyft.train(network2, xz2, n_steps = n_steps, lr = 1e-3, n_particles = n_particles)
losses += swyft.train(network2, xz2, n_steps = n_steps, lr = 1e-4, n_particles = n_particles)

xz3 = swyft.update_xz(xz2, network2, x0, model, n_sims = n_sims, lnL_th = np.log(1e-5), append=False)
network3 = swyft.MLP(x_dim, z_dim, n_hidden, xz_init = xz3)
losses += swyft.train(network3, xz3, n_steps = n_steps, lr = 1e-3, n_particles = n_particles)
losses += swyft.train(network3, xz3, n_steps = n_steps, lr = 1e-4, n_particles = n_particles)

# Plot results
z_lnL = swyft.estimate_lnL(network3, x0, swyft.get_z(xz3))
plt.plot(z_lnL[0]['z'], np.exp(z_lnL[0]['lnL']))
plt.axvline(0.1)
plt.axvline(0.9)
plt.savefig("figs/testrun_02.png")
