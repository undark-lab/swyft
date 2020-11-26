
[![Documentation Status](https://readthedocs.org/projects/swyft/badge/?version=latest)](https://swyft.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# SWYFT

Simulator efficient nested neural marginal posterior estimation

## Features

- The number of simulator runs required to estimate marginal posteriors is
  typically 10-100 times smaller than with sampling based methods.
- The number of nuisance parameters that need to be marginalized out can be
  very high (thousands, millions), without significantly affecting performance.

## Soon

- In contrast to Markov-chain based sampling methods, all simulator runs are
  stored, never rejected, and can be used for future analysis.
- SWYFT comes with an new stochastic simulation store based on Poisson point
  processes that enables the full reuse of previous simulator runs even after
  parameter transformations, change of priors, etc.
- We don't do Markov chains.  Simulations can run fully in parallel (via MPI).
- We can handle both parameter estimation as well as classification problems
  (odds ratio estimation).
- Multiple diagnostics to ensure that results are statistically sound.
- Battle-tested defaults.

## Caveats

- If data is complex (large images, volumina, etc), a hand-crafted neural
  network as featurization engine is necessary for optimal results.
- Life is eaiser if you have minimal experience with pytorch. A gpu makes 
  training the neural networks faster.

## Documentation

Documentation can be found [here](https://swyft.readthedocs.io/en/latest/).  
A quickstart example is avaliable on google colab! 
(Don't forget to use the gpu runtime!) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/undark-lab/swyft/blob/master/notebooks/QuickStart.ipynb)
