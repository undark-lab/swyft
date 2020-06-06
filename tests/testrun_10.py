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

x_dim = 100

# Definition of trivial test spectral model
def model(z, sigma = 0.1):
    grid = np.linspace(-10, 10, x_dim)
    y = z[0] * np.sin(grid)
    y = z[1] * np.sin(grid*1.1)
    n = np.random.randn(1) * sigma
    return y + n

# Generate test z0 and x0
z0 = np.array([0.30, 0.70])
x0 = model(z0, sigma = 0.1)
z_dim = len(z0)

# Initialize loss list
losses = []

# Generate 1-dim posteriors
xz = swyft.init_xz(model, n_sims = n_sims, n_dim = z_dim)
network = swyft.MLP(x_dim, z_dim, n_hidden, xz_init = xz)
losses += swyft.train(network, xz, n_steps = n_steps, lr = 1e-2, n_particles = n_particles)
losses += swyft.train(network, xz, n_steps = n_steps, lr = 1e-3, n_particles = n_particles)
losses += swyft.train(network, xz, n_steps = n_steps, lr = 1e-4, n_particles = n_particles)

# Generate 2-dim posteriors
network_2d = swyft.MLP_2d(x_dim, n_hidden, xz_init = xz)
losses += swyft.train(network_2d, xz, n_steps = n_steps, lr = 1e-2, n_particles = n_particles)
losses += swyft.train(network_2d, xz, n_steps = n_steps, lr = 1e-3, n_particles = n_particles)
losses += swyft.train(network_2d, xz, n_steps = n_steps, lr = 1e-4, n_particles = n_particles)

# Plot results 1-dim
z_lnL = swyft.estimate_lnL(network, x0, swyft.get_z(xz))
plt.clf()
plt.plot(z_lnL[0]['z'], np.exp(z_lnL[0]['lnL']))
plt.axvline(0.1)
plt.axvline(0.9)
plt.savefig("figs/testrun_10a.png")

# plot 2-dim results
z_lnL = swyft.estimate_lnL_2d(network_2d, x0, swyft.get_z(xz))
lnL = z_lnL['lnL']
z_ij = z_lnL['z']
plt.clf()
plt.tricontour([zzz[0] for zzz in z_ij], [zzz[1] for zzz in z_ij], lnL*2,
        levels = [-36, -25, -16, -9, -4, -1, 0])
t = np.linspace(0, 6.5, 1000)
xc = np.sin(t)*0.4+0.5
yc = np.cos(t)*0.4+0.5
plt.scatter(xc, yc)
plt.savefig('figs/testrun_10b.png')
