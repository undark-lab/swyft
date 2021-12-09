What is *swyft*?
================

Marginal posterior estimation
-----------------------------

*swyft* directly estimates marginal posteriors for parameters of interest
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


Nuisance parameters — Yes please!
---------------------------------

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

.. note::
   When performing likelihood-based modelling, it is sometimes necessary to
   simplify the model for computational purposes.
   *swyft* uses likelihood-free inference, which means that models can be as
   complex as they need to be to describe reality (more specifically, we use
   the effective AALR [1]).


Not a Markov Chain
------------------

*Likelihood-based techniques often use Markov chains*, which require a simulation
for every link in the chain. Due to the properties of Markov chains, it is not
possible to utilize those simulations again for further analysis.
That effort has been lost.

*Likelihood-free inference can be based on simulations that sample the
(constrained) prior*. Reusing these simulations is allowed, we don’t
have to worry about breaking the Markov chain.

.. note::
   *swyft* automatizes the re-use of simulator runs where appropriate, using a
   new resampling approach (iP3 sample caching [2]).


High precision
--------------

Likelihood-based techniques are highly precise by focusing simulator
runs on parameter space regions that are consistent with a particular
observation.

Likelihood-free inference techniques can be less precise when there are
too few simulations in parameter regions that matter most.

.. note::
   *swyft* uses a new nested sampling scheme to target parameter regions most
   relevant for a given observation. This allows similar precision to
   likelihood-based approaches, without the high number of simulator runs
   (nested ratio estimation, NRE [2]).


Where is the catch?
-------------------

*swyft* uses neural likelihood estimation. The package is supposed to work
out-of-the-box for simple low-dimensional data. However, tackling
complex and/or high-dimensional data (think of high-resolution images or
spectra, combination of multiple data sets) requires some basic skills
in writing neural networks using pytorch.

.. note::
   *swyft* SWYFT provides a simple gateway for spicing up your analysis
   with the power of neural network-based inference.


References
----------

[1] Joeri Hermans, Volodimir Begy, and Gilles Louppe. Likelihood-free mcmc
with amortized approximate ratio estimators. arXiv preprint arXiv:1903.04057, 2019.

[2] Benjamin Kurt Miller, Alex Cole, Gilles Louppe, and Christoph Weniger.
Simulation-efficient marginal posterior estimationwithswyft: stop wasting your freaking time.
arXiv preprint, 2020.
