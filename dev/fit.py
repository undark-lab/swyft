#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

import torch
import numpy as np
import pylab as plt
from omegaconf import OmegaConf

import swyft
import simulators
import heads

DEVICE = 'cuda:0'
CACHE_PATH = 'cache_FermiV1.zarr'

def noise(obs, params = None, sigma = 1.):
    data = {k: v+np.random.randn(*v.shape)*sigma for k, v in obs.items()}
    return data

def main():
    root = conf_cli=OmegaConf.from_cli().root
    conf = OmegaConf.load(root+".yaml")

    # Define model and prior
    prior = simulators.prior_FermiV1
    model = simulators.model_FermiV1

    # Target observation
    par0 = dict(ox=5., oy=5., a=1.5, p1=0.4, p2=1.1)
    obs0 = noise(model(par0))

    params = par0.keys()
    obs_shapes = {k: v.shape for k, v in obs0.items()}

    cache = swyft.DirectoryCache(params, obs_shapes = obs_shapes, path = CACHE_PATH)

    s = swyft.NestedRatios(model, prior, noise = None, obs = obs0, device = DEVICE,
            Ninit=conf.Ninit, Nmax = conf.Nmax, cache = cache)

    s.run(max_rounds=conf.max_rounds, train_args = conf.train_args, head = heads.Head_FermiV1, tail_args = conf.tail_args, head_args = conf.head_args)

    samples = s.marginals(obs0, 10000)
    swyft.plot.plot1d(samples, list(prior.params()), figsize = (20, 4), ncol = 5, grid_interpolate = True, truth = par0)
    plt.savefig("%s.marginals.pdf"%root)

    diagnostics = swyft.utils.sample_diagnostics(samples)
    state_dict = {
            "NestedRatios": s.state_dict(),
            "diagnostics": diagnostics
            }
    torch.save(state_dict, "%s.diags.pt"%root)


if __name__ == "__main__":
    main()
