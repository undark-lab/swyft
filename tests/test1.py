#!/usr/bin/env python3

import pylab as plt
import torch
import torch.nn as nn
import numpy as np
import swyft

# Definition of trivial test model
def model(z, sigma = 0.01):
    y = ((z[0]-0.5)**2 + (z[1]-0.5)**2)**0.5  # Radius
    n = np.random.randn(2) * sigma
    return y + n

# Generate test z0 and x0
z0 = np.array([0.10, 0.50])
x0 = model(z0)

# Fit model
sw = SWYFT(model, 2, x0, n_train = 10, n_sims = 10)
sw.run()
z_lnL = sw.get_post1d()

# Plot results
plt.plot(z_lnL[0]['z'], np.exp(z_lnL[0]['lnL']))
plt.axvline(0.1)
plt.axvline(0.9)
plt.savefig("test1.png")
