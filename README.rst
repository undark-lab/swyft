*swyft*
=======

.. image:: https://badge.fury.io/py/swyft.svg
   :target: https://badge.fury.io/py/swyft
   :alt: PyPI version


.. .. image:: https://github.com/undark-lab/swyft/actions/workflows/tests.yml/badge.svg
..    :target: https://github.com/undark-lab/swyft/actions
..    :alt: Tests


.. .. image:: https://github.com/undark-lab/swyft/actions/workflows/syntax.yml/badge.svg
..    :target: https://github.com/undark-lab/swyft/actions
..    :alt: Syntax


.. image:: https://codecov.io/gh/undark-lab/swyft/branch/master/graph/badge.svg?token=E253LRJWWE
   :target: https://codecov.io/gh/undark-lab/swyft
   :alt: codecov


.. .. image:: https://readthedocs.org/projects/swyft/badge/?version=latest
..    :target: https://swyft.readthedocs.io/en/latest/?badge=latest
..    :alt: Documentation Status


.. .. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
..    :target: https://github.com/undark-lab/swyft/blob/master/CONTRIBUTING.md
..    :alt: Contributions welcome


.. .. image:: https://colab.research.google.com/assets/colab-badge.svg
..    :target: https://colab.research.google.com/github/undark-lab/swyft/blob/master/notebooks/Quickstart.ipynb
..    :alt: colab

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5752735.svg
   :target: https://doi.org/10.5281/zenodo.5752735

*swyft* is the official implementation of Truncated Marginal Neural Ratio Estimation (TMNRE),
a hyper-efficient, simulation-based inference technique for complex data and expensive simulators.

* **Documentation & installation**: https://swyft.readthedocs.io/en/latest/
* **Example usage**: https://swyft.readthedocs.io/en/latest/tutorial-notebooks.html
* **Source code**: https://github.com/undark-lab/swyft
* **Support & discussion**: https://github.com/undark-lab/swyft/discussions
* **Bug reports**: https://github.com/undark-lab/swyft/issues
* **Contributing**: https://github.com/undark-lab/swyft/blob/master/CONTRIBUTING.md
* **Citation**: https://swyft.readthedocs.io/en/latest/citation.html

*swyft*:

* estimates likelihood-to-evidence ratios for arbitrary marginal posteriors; they typically require fewer simulations than the corresponding joint.
* performs targeted inference by prior truncation, combining simulation efficiency with empirical testability.
* seamless reuses simulations drawn from previous analyses, even with different priors.
* integrates `dask <https://dask.org/>`_ and `zarr <https://zarr.readthedocs.io/en/stable/>`_ to make complex simulation easy.

*swyft* is designed to solve the Bayesian inverse problem when the user has access to a simulator that stochastically maps parameters to observational data.
In scientific settings, a cost-benefit analysis often favors approximating the posterior marginality; *swyft* provides this functionality.
The package additionally implements our prior truncation technique, routines to empirically test results by estimating the expected coverage,
and a `dask <https://dask.org/>`_ simulator manager with `zarr <https://zarr.readthedocs.io/en/stable/>`_ storage to simplify use with complex simulators.



Related
-------

* `tmnre <https://github.com/bkmi/tmnre>`_ is the implementation of the paper `Truncated Marginal Neural Ratio Estimation <https://arxiv.org/abs/2107.01214>`_.
* `v0.1.2 <https://github.com/undark-lab/swyft/releases/tag/v0.1.2>`_ is the implementation of the paper `Simulation-efficient marginal posterior estimation with swyft: stop wasting your precious time <https://arxiv.org/abs/2011.13951>`_.
* `sbi <https://github.com/mackelab/sbi>`_ is a collection of simulation-based inference methods. Unlike *swyft*, the repository does not include truncation nor marginal estimation of posteriors.
