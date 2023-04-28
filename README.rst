Swyft
=====

.. image:: https://raw.githubusercontent.com/undark-lab/swyft/v0.4.1/docs/source/_static/img/swyft_logo_wide.png
   :width: 800
   :align: center

*Swyft* is a system for scientific simulation-based inference at scale.

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


.. image:: https://joss.theoj.org/papers/10.21105/joss.04205/status.svg
   :target: https://doi.org/10.21105/joss.04205


.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5752734.svg
   :target: https://doi.org/10.5281/zenodo.5752734


*Swyft* is the official implementation of Truncated Marginal Neural Ratio Estimation (TMNRE),
a hyper-efficient, simulation-based inference technique for complex data and expensive simulators.


Swyft in action
---------------


.. image:: https://raw.githubusercontent.com/undark-lab/swyft/v0.4.1/docs/source/_static/img/SBI-curve.gif
   :width: 800
   :align: center



* Swyft makes it convenient to perform Bayesian or Frequentist inference of hundreds, thousands or millions of parameter posteriors by constructing optimal data summaries. 
* To this end, Swyft estimates likelihood-to-evidence ratios for arbitrary marginal posteriors; they typically require fewer simulations than the corresponding joint.
* Swyft performs targeted inference by prior truncation, combining simulation efficiency with empirical testability.
* Swyft is based on stochastic simulators, which map parameters stochastically to observational data. Swyft makes it convenient to define such simulators as graphical models.
* In scientific settings, a cost-benefit analysis often favors approximating the posterior marginality; *swyft* provides this functionality.
* The package additionally implements our prior truncation technique, routines to empirically test results by estimating the expected coverage, and a simulator manager with `zarr <https://zarr.readthedocs.io/en/stable/>`_ storage to simplify use with complex simulators.


Papers using Swyft/TMNRE
------------------------

2021

- “Fast and Credible Likelihood-Free Cosmology with Truncated Marginal Neural Ratio Estimation“ Cole+ https://arxiv.org/abs/2111.08030

2022

- “Estimating the warm dark matter mass from strong lensing images with truncated marginal neural ratio estimation” Anau Montel+, https://arxiv.org/abs/2205.09126
- “SICRET: Supernova Ia Cosmology with truncated marginal neural Ratio EsTimation” Karchev+ https://arxiv.org/abs/2209.06733
- “One never walks alone: the effect of the perturber population on subhalo measurements in strong gravitational lenses” Coogan+ https://arxiv.org/abs/2209.09918
- “Detection is truncation: studying source populations with truncated marginal neural ratio estimation” Anau Montel+ https://arxiv.org/abs/2211.04291

2023

- “Debiasing Standard Siren Inference of the Hubble Constant with Marginal Neural Ratio Estimation” Gagnon-Hartman+ https://arxiv.org/abs/2301.05241
- “Constraining the X-ray heating and reionization using 21-cm power spectra with Marginal Neural Ratio Estimation” Saxena+ https://arxiv.org/abs/2303.07339
- “Peregrine: Sequential simulation-based inference for gravitational wave signals”, Bhardwaj+ https://arxiv.org/abs/2304.02035
- “Albatross: A scalable simulation-based inference pipeline for analysing stellar streams in the Milky Way”, Alvey+ https://arxiv.org/abs/2304.02032


Further information
-------------------

* **Documentation & installation**: https://swyft.readthedocs.io/
* **Example usage**: https://swyft.readthedocs.io/en/latest/tutorial-notebooks.html
* **Source code**: https://github.com/undark-lab/swyft
* **Support & discussion**: https://github.com/undark-lab/swyft/discussions
* **Bug reports**: https://github.com/undark-lab/swyft/issues
* **Contributing**: https://swyft.readthedocs.io/en/latest/contributing-link.html
* **Citation**: https://swyft.readthedocs.io/en/latest/citation.html


*Swyft* history
---------------

* As of v0.4.0, *Swyft* is based on pytorch-lightning, with a completely updated
* `v0.3.2 <https://github.com/undark-lab/swyft/releases/tag/v0.3.2>`_ is the version that was submitted to `JOSS <https://joss.theoj.org/papers/10.21105/joss.04205>`_.
* `tmnre <https://github.com/bkmi/tmnre>`_ is the implementation of the paper `Truncated Marginal Neural Ratio Estimation <https://arxiv.org/abs/2107.01214>`_.
* `v0.1.2 <https://github.com/undark-lab/swyft/releases/tag/v0.1.2>`_ is the implementation of the paper `Simulation-efficient marginal posterior estimation with swyft: stop wasting your precious time <https://arxiv.org/abs/2011.13951>`_.

Relevant packages
-----------------

* `sbi <https://github.com/mackelab/sbi>`_ is a collection of simulation-based inference methods. Unlike *Swyft*, the repository does not include our truncation scheme nor marginal estimation of posteriors.

* `lampe <https://github.com/francois-rozet/lampe>`_ is an implementation of amoritzed simulation-based inference methods aimed at simulation-based inference researchers due to its flexibility.
