Why SWYFT?
==========

SWYFT estimates marginal posteriors for parameters of interest
:math:`\mathbf{z}`, which are formally obtained by integrating over all
remaining (nuisance) parameters :math:`\boldsymbol{\eta}`,

.. math:: p(\mathbf{z}|\mathbf{x}) = \frac{\int d\boldsymbol{\eta}\, p(\mathbf{x}|\mathbf{z}, \boldsymbol{\eta}) p(\boldsymbol{\eta}) p(\mathbf{z})}{p(\mathbf{x})}\;.

Here, :math:`p(\mathbf{x}|\mathbf{z}, \boldsymbol{\eta})` is our forward
model that includes the includes both the physics and detector
simulator, :math:`p(\mathbf{z})` and :math:`p(\boldsymbol{\eta})` are
parameter priors, and :math:`p(\mathbf{x})` is the Bayesian evidence.

SWYFT aims at combining the flexibility and power of likelihood-free
with the convenience and precision of likelihood-based inference. It
follows the formula SWYFT = AALR + NRE + iP3.

Nuisance parameters — Yes please!
---------------------------------

*In the context of likelihood-based inference, nuisance parameters are
an integration problem.* Given the likelihood density
:math:`p(\mathbf{x}|\mathbf{z}, \boldsymbol{\eta})` for a particular
observation :math:`\mathbf{x}`, one attempts to solve the above integral
over :math:`\boldsymbol{\eta}`, e.g. through sampling based methods.
This becomes increasingly challenging if the number of nuisance
parameters grows.

*In the context of likelihood-free inference, nuisance parameters are
noise.* Posteriors are estimated based on a large number of training
samples
:math:`\mathbf{x}, \mathbf{z}\sim p(\mathbf{x}|\mathbf{z}, \boldsymbol{\eta})p(\mathbf{z})p(\boldsymbol{\eta})`,
no matter the dimension of the nuisance parameter space. For a given
:math:`\mathbf{z}`, more nuisance parameters just increase the variance
of :math:`\mathbf{x}` (which oddly enough can make the inference problem
simpler rather than more difficult).

SWYFT uses likelihood-free inference, which means that models can be as
complex as they need to be to describe reality (more specifically, we
use the effective AALR [1]).

Not a Markov Chain
------------------

*Likelihood-based techniques often use Markov chains*, which require
simulations tailor made for each particular analysis. Those simulations
are lost afterwards.

*Likelihood-free inference can be based on simulations that sample the
(constrained) prior*. Reusing these simulations is allowed, we don’t
have to worry about breaking the Markov chain.

SWYFT automatizes the re-use of simulator runs where appropriate, using
a new resampling approach (iP3 sample caching [2]).

High precision
--------------

Likelihood-based techniques are highly precise by focusing simulator
runs on parameter space regions that are consistent with a particular
observation.

Likelihood-free inference techniques can be less precise when there are
too few simulations in parameter regions that matter most.

SWYFT uses a new nested sampling scheme to target parameter regions most
relevant for a given observation. This allows similar precision to
likelihood-based approaches, without the high number of simulator runs
(nested ratio estimation, NRE [2]).

Where is the catch?
-------------------

SWYFT uses neural likelihood estimation. The package is supposed to work
out-of-the-box for simple low-dimensional data. However, tackling
complex and/or high-dimensional data (think of high-resolution images or
spectra, combination of multiple data sets) requires some basic skills
in writing neural networks using pytorch.

SWYFT provides a simple gateway for spicing up your analysis with the
power of neural network-based inference.

Our aspirational goals
----------------------

We developed SWYFT to solve (our) real-world data analysis problems. We
have the following goals in mind, which are already partially realized.

-  SWYFT is meant as a drop-in replacement for sampling- and
   likelihood-based methods like, e.g., MultiNest.
-  SWYFT should be in particular useful in the context of very expensive
   and slow simulators.
-  SWYFT is efficient initially and enables hyper-efficient follow up
   studies.
-  …

We are currently developing SWYFT towards a full-fledged reliable
inference tool, together with engineers the Dutch eScience center and
SURFsara, as part of the eTEC-BIG grant “Dark Generators”. Any feedback
will be useful to shape the development of this tool. Stay tuned!

How does it work?
-----------------

Details can be found here:

`SWYFT theory
documentation <https://www.notion.so/SWYFT-theory-documentation-061804b34f0447178a5904617cf76745>`__

References
----------

[1] AALR

[2] SWYFT paper
