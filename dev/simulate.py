#!/usr/bin/env python3

import sys

import simulators
import torch


def main():
    try:
        infile = sys.argv[1]
        outfile = sys.argv[2]
    except:
        print("Usage: simulate.py infile outfile")
    model = simulators.model_FermiV1
    params = torch.load(infile)
    obs = model(params)
    torch.save(obs, outfile)


if __name__ == "__main__":
    main()
