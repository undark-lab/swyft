#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pylab as plt
import swyft
import torch

DEVICE = 'cuda:0'

# A toy model with a diffuse and point source component
def model(z, sigma = .1):
    X, Y = np.mgrid[-5:5:50j, -5:5:50j]
    
    # parameters z[0] ... z[2] determine diffuse bkg
    iso = z[0] * np.ones_like(X)
    scale = z[2] + 0.1
    disk = 10 * z[1] * np.exp(-X**2/scale**2/2)
    diff = disk + iso
    
    # N point sources
    N = int(z[4] * 10000)
    # spatial distribution uniform
    i, j = np.random.choice(50, (2, N))  # just indices
    # log normal distribution for fluxes, depends on z[3]
    f = 10**(np.random.randn(N)*0.5) * z[3]
    
    # point source map
    psc = np.zeros_like(X)
    psc[i, j] += f
    
    # Gamma-ray sky
    sky = psc + diff
    
    return np.array(sky)


def noisemodel(x, z, sigma = 0.1):
    x = x.copy()
    x[1] = x[1] + np.random.randn(*x[1].shape)*sigma
    return x


def get_mock_obs():
    z0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    x0 = noisemodel(model(z0), z0)
    #plt.imshow(x0)
    return x0, z0


# Convolutional network as HEAD of inference network
class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0a = torch.nn.Conv2d(1, 100, 1)
        self.conv0b = torch.nn.Conv2d(100, 1, 1)
        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv2 = torch.nn.Conv2d(10, 20, 5)
        self.conv3 = torch.nn.Conv2d(20, 40, 5)
        self.pool = torch.nn.MaxPool2d(2)
        
    def forward(self, x):
        nbatch = len(x)
        
        x = x.unsqueeze(1)
        x = self.conv0a(x)
        x = torch.relu(x)
        x = self.conv0b(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(nbatch, -1)

        return x

NROUNDS = 3
MAX_EPOCHS = 20
NSAMPLES = 1000

def main():
    x0, z0 = get_mock_obs()
    zdim = len(z0)
    comb1d = [[i] for i in range(zdim)]

    # Instantiate datastore
    ds = swyft.DataStore().init(zdim = zdim, xdim = x0.shape)

    # Analysis rounds 1-dim posteriors
    re_prev = None
    for i in range(NROUNDS):
        td = swyft.TrainData(x0 = x0, zdim = zdim, noisehook = noisemodel, datastore = ds, nsamples = NSAMPLES, threshold = 1e-5, parent = re_prev)
        ds.simulate(model)
        re = swyft.RatioEstimation(zdim, td, device = DEVICE, combinations = comb1d, head = Head)
        tl, vl = re.train(max_epochs = MAX_EPOCHS, nbatch = 32, lr_schedule = [1e-3, 5e-4])
        re_prev = re

    final_validation_loss_1dim = vl[-1][-1]

    # Analysis rounds 2-dim posteriors
    re2 = swyft.RatioEstimation(zdim, td, device = DEVICE, combinations = swyft.comb2d([0, 1, 2, 3, 4]), head = Head)
    tl, vl = re2.train(max_epochs = MAX_EPOCHS, nbatch = 32, early_stopping_patience = 3, lr_schedule = [1e-3, 5e-4])

    final_validation_loss_2dim = vl[-1][-1]

    #swyft.plot1d(re, x0 = x0, z0 = z0, cmap = 'Greys', dims = (25, 3), ncol = 5)
    #swyft.corner(re, re2, x0 = x0, z0 = z0, cmap = 'Greys', dim = 15, Nmax = 10000)


    # Optimization metrics
    print("Final validation loss 1-dim posteriors", final_validation_loss_1dim)
    print("Final validation loss 2-dim posteriors", final_validation_loss_2dim)


if __name__ == "__main__":
    main()
