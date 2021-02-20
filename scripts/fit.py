#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.DEBUG, format="%(message)s")

import numpy as np
import pylab as plt
import torch
from omegaconf import OmegaConf

import swyft
from swyft_model import noise, prior, model, par0, obs0, CustomHead

DEVICE = "cuda:0"

def main():
    conf = OmegaConf.load("swyft_config.yaml")

    params = par0.keys()
    obs_shapes = {k: v.shape for k, v in obs0.items()}
    cache = swyft.DirectoryCache(params, obs_shapes=obs_shapes, path=conf.cache)

    s = swyft.NestedRatios(
        model,
        prior,
        noise=noise,
        obs=obs0,
        device=DEVICE,
        Ninit=conf.Ninit,
        Nmax=conf.Nmax,
        cache=cache,
    )

    s.run(
        max_rounds=conf.max_rounds,
        train_args=conf.train_args,
        head=CustomHead,
        tail_args=conf.tail_args,
        head_args=conf.head_args,
    )

    samples = s.marginals(obs0, 3000)
    swyft.plot.plot1d(
        samples,
        list(prior.params()),
        figsize=(20, 4),
        ncol=5,
        grid_interpolate=True,
        truth=par0,
    )
    plt.savefig("marginals.pdf")

    diagnostics = swyft.utils.sample_diagnostics(samples)
    state_dict = {"NestedRatios": s.state_dict(), "diagnostics": diagnostics}
    torch.save(state_dict, "sample_diagnostics.pt")

if __name__ == "__main__":
    main()
