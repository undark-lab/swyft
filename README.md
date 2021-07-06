[![PyPI version](https://badge.fury.io/py/swyft.svg)](https://badge.fury.io/py/swyft)
[![Tests](https://github.com/undark-lab/swyft/actions/workflows/tests.yml/badge.svg)](https://github.com/undark-lab/swyft/actions)
[![Syntax](https://github.com/undark-lab/swyft/actions/workflows/syntax.yml/badge.svg)](https://github.com/undark-lab/swyft/actions)
[![codecov](https://codecov.io/gh/undark-lab/swyft/branch/master/graph/badge.svg?token=E253LRJWWE)](https://codecov.io/gh/undark-lab/swyft)
[![Documentation Status](https://readthedocs.org/projects/swyft/badge/?version=latest)](https://swyft.readthedocs.io/en/latest/?badge=latest)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/undark-lab/swyft/blob/master/CONTRIBUTING.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Check out the quickstart notebook --> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/undark-lab/swyft/blob/master/notebooks/Quickstart.ipynb)

**Disclaimer: swyft is research software under heavy development and still in its alpha phase. There are many rough edges, and things might break. However, the core algorithms work, and we use swyft in production for research papers. If you encounter problems, please contact the authors or submit a bug report.**

# SWYFT

<p align="center">
Neural nested marginal posterior estimation
</p>

*Cursed by the dimensionality of your nuisance space? Wasted by Markov
chains that reject your simulations? Exhausted from messing with
simplistic models, because your inference algorithm cannot handle the
truth? Try swyft for some pain relief.*

A simple example is avaliable on [google colab](https://colab.research.google.com/github/undark-lab/swyft/blob/master/notebooks/Quickstart.ipynb).

Our repository applying swyft to benchmarks and example inference problems is available at [tmnre](https://github.com/bkmi/tmnre).

## Installation

**After installing [pytorch](https://pytorch.org/get-started/locally/)**, please run the command:

`pip install swyft`

## Relevant Tools

swyft exists in an ecosystem of posterior estimators. The project [sbi](https://github.com/mackelab/sbi) is particularly relevant as it is a collection of likelihood-free / simulator-based methods.


## Citing

If you use *swyft* in scientific publications, please cite:

*Truncated Marginal Neural Ratio Estimation*. Benjamin Kurt Miller, Alex Cole, Patrick Forré, Gilles Louppe, Christoph Weniger. https://arxiv.org/abs/2107.01214

*Simulation-efficient marginal posterior estimation with swyft: stop wasting your precious time*. Benjamin Kurt Miller, Alex Cole, Gilles Louppe, Christoph Weniger. https://arxiv.org/abs/2011.13951
