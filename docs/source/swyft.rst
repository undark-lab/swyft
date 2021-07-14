Why *swyft*?
============

Overview
--------

With *swyft* our goal is to provide a general, flexible, reliable and practical
tool for solving hard Bayesian parameter inference problems in physics and
astronomy.  *swyft* uses a specific flavor of simulation-based neural inference
techniques (truncated marginal neural ratio estimation [1]), that offers
multiple advantages over established Markov Chain based methods, or other
simulation-based neural approaches.

- *swyft* directly estimates marginal posteriors, which typically requires far
  less simulation runs than estimating the full joint posterior.
- *swyft* uses a simulation store that make re-use of simulations, even with
  different priors, efficient and seamless.
- *swyft* performs targeted inference by prior truncation, which combines the
  simulation efficiency of existing sequential methods with the testability of
  amortized methods.

Details
-------

Marginal posterior estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*swyft* can directly estimate marginal posteriors for parameters of interest
:math:`\mathbf{z}`, given some observation :math:`\mathbf{x}`. These are
formally obtained by integrating over all remaining (nuisance) parameters
:math:`\boldsymbol{\eta}`,

.. math::
   p(\mathbf{z}|\mathbf{x}) = \frac{\int d\boldsymbol{\eta}\,
   p(\mathbf{x}|\mathbf{z}, \boldsymbol{\eta}) p(\boldsymbol{\eta}, \mathbf{z})}
   {p(\mathbf{x})}\;.

Here, :math:`p(\mathbf{x}|\mathbf{z}, \boldsymbol{\eta})` is an abritrary
forward model that includes both the physics and detector simulator,
:math:`p(\mathbf{z}, \boldsymbol{\eta})` is the joint prior,
and :math:`p(\mathbf{x})` is the Bayesian evidence.


Nuisance parameters
^^^^^^^^^^^^^^^^^^^

*In the context of likelihood-based inference, nuisance parameters are an
integration problem.* Given the likelihood density
:math:`p(\mathbf{x}|\mathbf{z}, \boldsymbol{\eta})` for a particular
observation :math:`\mathbf{x}`, one attempts to solve the above integral over
:math:`\boldsymbol{\eta}`, e.g. through sampling based methods.  This becomes
increasingly challenging if the number of nuisance parameters grows.

*In the context of likelihood-free inference, nuisance parameters are noise.*
Posteriors are estimated based on a large number of training samples
:math:`\mathbf{x}, \mathbf{z}\sim p(\mathbf{x}|\mathbf{z},
\boldsymbol{\eta})p(\mathbf{z}, \boldsymbol{\eta})`, no matter the dimension
of the nuisance parameter space. For a given :math:`\mathbf{z}`, more nuisance
parameters just increase the variance of :math:`\mathbf{x}` (which oddly enough
can make the inference problem simpler rather than more difficult).


Simulation re-use
^^^^^^^^^^^^^^^^^

*Likelihood-based techniques often use Markov chains*, which require a simulation
for every link in the chain. Due to the properties of Markov chains, it is not
possible to utilize those simulations again for further analysis.
That effort has been lost.

*Likelihood-free inference can be based on simulations that sample the
(constrained) prior*. Reusing these simulations is allowed, we don’t
have to worry about breaking the Markov chain.


High precision through targeted inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Likelihood-based techniques are highly precise by focusing simulator
runs on parameter space regions that are consistent with a particular
observation.

Likelihood-free inference techniques can be less precise when there are
too few simulations in parameter regions that matter most.


Learned features
^^^^^^^^^^^^^^^^

*swyft* uses neural likelihood estimation. The package works out-of-the-box for
low-dimensional data.  Tackling complex and/or high-dimensional data (e.g.,
high-resolution images or spectra, combination of multiple data sets) is
possible through providing custom feature extractor networks in pytorch.


References
----------

[1] Joeri Hermans, Volodimir Begy, and Gilles Louppe. Likelihood-free mcmc
with amortized approximate ratio estimators. arXiv preprint arXiv:1903.04057, 2019.

[2] Benjamin Kurt Miller, Alex Cole, Gilles Louppe, and Christoph Weniger.
Simulation-efficient marginal posterior estimationwithswyft: stop wasting your freaking time.
arXiv preprint, 2020.
