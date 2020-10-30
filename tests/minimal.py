#!/usr/bin/env python3

import matplotlib as mpl

mpl.use("Agg")
import pylab as plt
import torch
import torch.nn as nn
import numpy as np
import swyft

N_TRAIN = 10000
N_SIMS = 10000


def model(z, sigma=0.05):
    """Measurement of radial distance from z_c."""
    r = ((z - 0.5) ** 2).sum() ** 0.5  # Radial distance from z_c = (0.5, 0.5, ...)
    n = np.random.randn(1) * sigma
    return r + n


# Generate true x0 and z0
z0 = np.array([0.10, 0.50])
x0 = model(z0, sigma=0.0)  # Asimov data

# Fit model
sw = swyft.SWYFT(model, 2, x0, n_train=N_TRAIN, n_sims=N_SIMS)
sw.run()
z_lnL = sw.get_post1d()

# Plot results
for i in range(len(z0)):
    plt.plot(z_lnL[i]["z"], np.exp(z_lnL[i]["lnL"]))
plt.axvline(0.1, ls="--", color="0.5")
plt.axvline(0.9, ls="--", color="0.5")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig("figs/minimal.png")
