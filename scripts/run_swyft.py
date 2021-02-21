#!/usr/bin/env python3
# Requires python 3.5+

import logging
logging.basicConfig(level=logging.DEBUG, format="%(message)s")

import os
import importlib.util
import numpy as np
import pylab as plt
import torch
from omegaconf import OmegaConf

import swyft

DEVICE = "cuda:0"

def main():
    # Pretty hacky way to import local model
    # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    cwd = os.getcwd()
    spec = importlib.util.spec_from_file_location("defs", cwd+"/definitions.py")
    defs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(defs)

    # Load configuration file
    conf = OmegaConf.load("config.yaml")

    # Set up cache
    params = defs.par0.keys()
    obs_shapes = {k: v.shape for k, v in defs.obs0.items()}
    cache = swyft.DirectoryCache(params, obs_shapes=obs_shapes, path=conf.cache)

    # Set up nested ratio estimator
    s = swyft.NestedRatios(
        defs.model,
        defs.prior,
        noise=defs.noise,
        obs=defs.obs0,
        device=DEVICE,
        Ninit=conf.Ninit,
        Nmax=conf.Nmax,
        cache=cache,
    )

    # Fit!
    s.run(
        max_rounds=conf.max_rounds,
        train_args=conf.train_args,
        head=defs.CustomHead,
        tail_args=conf.tail_args,
        head_args=conf.head_args,
    )

    # Post processing and evaluation
    samples = s.marginals(defs.obs0, 3000)

    # Save marginals
    swyft.plot.plot1d(
        samples,
        list(defs.prior.params()),
        figsize=(20, 4),
        ncol=5,
        grid_interpolate=True,
        truth=defs.par0,
    )
    plt.savefig("marginals.pdf")

    # Save diagnostics
    diagnostics = swyft.utils.sample_diagnostics(samples)
    state_dict = {"NestedRatios": s.state_dict(), "diagnostics": diagnostics}
    torch.save(state_dict, "sample_diagnostics.pt")

if __name__ == "__main__":
    main()
