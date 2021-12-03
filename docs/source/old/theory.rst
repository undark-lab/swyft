Theoretical concepts
====================

Introduction
============

Parametric stochastic simulators are ubiquitous in the physical sciences
:raw-latex:`\cite{Banik_2018, Bartels_2016, Rodr_guez_Puebla_2016}`.
However, performing parameter inference based on simulator runs using
Markov chain Monte Carlo is inconvenient or even impossible if the model
parameter space is large or the likelihood function is intractable. This
problem is addressed by so-called likelihood-free inference
:raw-latex:`\cite{sisson2018handbook}` or simulation-based inference
:raw-latex:`\cite{Cranmer2020}` techniques. Deep learning based
likelihood-free inference algorithms were organized into a taxonomy in
Ref. :raw-latex:`\cite{Durkan2020}`, where methods that estimated
likelihood ratios in a series of rounds were denoted Sequential Ratio
Estimation (SRE) :raw-latex:`\cite{Hermans2019}`. Our presented method
is closely related.

We propose *Nested Ratio Estimation* (NRE), which approximates the
likelihood-to-evidence ratio in a sequence of rounds. Loosely inspired
by the contour sorting method of nested sampling
:raw-latex:`\cite{Skilling2006, Feroz2008, Handley2015}`, the scheme
alternates between sampling from a constrained prior and estimating
likelihood-to-evidence ratios. It allows for efficient estimation of any
marginal posteriors of interest. Furthermore, we propose an algorithm
that we call *iP3 sample caching*, which facilitates simulator
efficiency by automatizing the reuse of previous simulator runs through
resampling of cached simulations.

The primary use case for these algorithms is the calculation of
arbitrary, low-dimensional marginal posteriors, typically in one or two
dimensions. In physics and astronomy, such marginals serve as the basis
for scientific conclusions by constraining individual model parameters
within uncertainty bounds. We implement a multi-target training regime
where all marginal posteriors of interest can be learned simultaneously.
We find that learning is simplified when one calculates each marginal
distribution directly rather than computing the full joint posterior and
marginalizing numerically. Furthermore, the method facilitates
effortless marginalization over arbitrary numbers of nuisance
parameters, increasing its utility in high-dimensional parameter
regimes–even to simulators with a tractable, yet high-dimensional,
likelihood :raw-latex:`\cite{lensing}`.

Nested Ratio Estimation (NRE).
==============================

We operate in the context of simulation-based inference where our
simulator :math:`\mathbf{g}` is a nonlinear function mapping a vector of
parameters
:math:`\boldsymbol{\theta}= (\theta_{1}, \dots, \theta_{d}) \in \mathbb{R}^{d}`
and a stochastic latent state :math:`\mathbf{z}` to an observation
:math:`\mathbf{x}= \mathbf{g}(\boldsymbol{\theta}, \mathbf{z})`. The
likelihood function is therefore
:math:`p(\mathbf{x}\vert \boldsymbol{\theta}) = \int \delta(\mathbf{x}- \mathbf{g}(\boldsymbol{\theta}, \mathbf{z})) \, p(\mathbf{z}\vert \boldsymbol{\theta}) \, d\mathbf{z}`,
with :math:`\delta(\cdot)` denoting the Dirac delta. Consider a
factorizable prior
:math:`p(\boldsymbol{\theta}) = p(\theta_{1}) \cdots p(\theta_{d})` over
the parameters, the joint posterior is given via Bayes’ rule as
:math:`p(\boldsymbol{\theta}|\mathbf{x}) = p(\mathbf{x}|\boldsymbol{\theta})p(\boldsymbol{\theta})/p(\mathbf{x})`,
where :math:`p(\mathbf{x})` is the evidence.

Our goal is to compute the marginal posterior,
:math:`p(\boldsymbol{\vartheta}\vert \mathbf{x})`, where
:math:`\boldsymbol{\vartheta}` are the parameters of interest. We will
denote all other parameters by :math:`\boldsymbol{\eta}`, such that
:math:`\boldsymbol{\theta}= (\boldsymbol{\vartheta}, \boldsymbol{\eta})`.
The marginal posterior is obtained from the joint distribution
:math:`p(\boldsymbol{\vartheta}, \boldsymbol{\eta}|\mathbf{x}) \equiv p(\boldsymbol{\theta}|\mathbf{x})`
by integrating over all components of :math:`\boldsymbol{\eta}`,

.. math::

   \label{eqn:post}
   p(\boldsymbol{\vartheta}\vert \mathbf{x})  \equiv \int p(\boldsymbol{\vartheta}, \boldsymbol{\eta}| \mathbf{x}) d\boldsymbol{\eta}
   = \int \frac{p(\mathbf{x}| \boldsymbol{\vartheta}, \boldsymbol{\eta})}{p(\mathbf{x})}
   p(\boldsymbol{\theta})
   %\prod_{j \notin \texttt{idx}} d\theta_{j}
   d\boldsymbol{\eta}
   = \frac{p(\mathbf{x}|\boldsymbol{\vartheta})}{p(\mathbf{x})}p(\boldsymbol{\vartheta})\;,

where we used Bayes’ rule and defined the marginal likelihood
:math:`p(\mathbf{x}|\boldsymbol{\vartheta})` in the last step.

Just like in SRE, we focus on a specific observation of interest,
:math:`\mathbf{x}_0`. Only parameter values :math:`\boldsymbol{\theta}`
that could have plausibly generated observation :math:`\mathbf{x}_0`
will significantly contribute to the integrals in
Eq. `[eqn:post] <#eqn:post>`__. For implausible values the likelihood
:math:`p(\mathbf{x}_0|\boldsymbol{\theta})` will be negligible. We
denote priors that are suitably constrained to plausible parameter
values by :math:`\tilde{p}(\theta_1, \dots, \theta_d)`. Similarly,
:math:`\tilde{\square}` indicates quantities :math:`\square` that are
calculated using the constrained prior. Therefore, using a judiciously
chosen constrained prior, accurately approximates the marginal posterior
in place of our true prior beliefs,

.. math::

   p(\boldsymbol{\vartheta}| \mathbf{x}_0) =
   \frac{p(\mathbf{x}_0|\boldsymbol{\vartheta})}{p(\mathbf{x}_0)} p(\boldsymbol{\vartheta}) \simeq
   \frac{\tilde{p}(\mathbf{x}_0|\boldsymbol{\vartheta})}{\tilde{p}(\mathbf{x}_0)} \tilde{p}(\boldsymbol{\vartheta})\;.

The increased probability that constrained priors assign to the
plausible parameter region cancels when dividing by the constrained
evidence :math:`\tilde p(\mathbf{x})`. We define the marginal
likelihood-to-evidence ratio

.. math::

   \label{eqn:likelihood_ratio}
       \tilde{r}(\mathbf{x}, \boldsymbol{\vartheta})
       \equiv \frac{\tilde{p}(\mathbf{x}\vert \boldsymbol{\vartheta})}{\tilde{p}(\mathbf{x})}
       = \frac{\tilde{p}(\mathbf{x}, \boldsymbol{\vartheta})}{\tilde{p}(\mathbf{x}) \tilde{p}(\boldsymbol{\vartheta})}
       = \frac{\tilde{p}(\boldsymbol{\vartheta}\vert\mathbf{x})}{\tilde{p}(\boldsymbol{\vartheta})}\;,

which is sufficient to evaluate the marginal posterior in
Eq. `[eqn:post] <#eqn:post>`__, and which we will now estimate. Under
the assumption of equal class population, it is known
:raw-latex:`\cite{Durkan2020, Cranmer2015}` that one can recover density
ratios using binary classification to distinguish between samples from
two distributions. Our binary classification problem is to distinguish
positive samples,
:math:`(\mathbf{x}, \boldsymbol{\vartheta}) \sim \tilde{p}(\mathbf{x}, \boldsymbol{\vartheta}) = p(\mathbf{x}\vert \boldsymbol{\vartheta}) \tilde{p}(\boldsymbol{\vartheta})`,
drawn jointly, and negative samples,
:math:`(\mathbf{x}, \boldsymbol{\vartheta}) \sim \tilde{p}(\mathbf{x}) \tilde{p}(\boldsymbol{\vartheta})`,
drawn marginally. The binary classifier
:math:`\sigma(f_{\phi}(\mathbf{x}, \boldsymbol{\vartheta}))` performs
optimally when
:math:`f_{\phi}(\mathbf{x}, \boldsymbol{\vartheta}) = \log \tilde{r}(\mathbf{x}, \boldsymbol{\vartheta})`,
where :math:`\sigma(\cdot)` is the sigmoid function and :math:`f_{\phi}`
is a neural network parameterized by :math:`\phi`. The associated binary
cross-entropy loss function used to train the ratio
:math:`\tilde{r}(\boldsymbol{\vartheta}, \mathbf{x}_0)` via stochastic
gradient descent is given by

.. math:: \ell = -\int \left[ \tilde{p}(\mathbf{x}|\boldsymbol{\vartheta})\tilde{p}(\boldsymbol{\vartheta}) \ln \sigma(f_\phi(\mathbf{x}, \boldsymbol{\vartheta})) + \tilde{p}(\mathbf{x})\tilde{p}(\boldsymbol{\vartheta}) \ln \sigma(-f_\phi(\mathbf{x},\boldsymbol{\vartheta})) \right] d\mathbf{x}\, d\boldsymbol{\vartheta}\;.

We propose to iteratively improve marginal posterior estimates in
:math:`R` rounds by employing posterior estimates from previous rounds
to define constrained priors. In each round :math:`r`, we estimate *all*
1-dim marginal posteriors, using :math:`d` instances of the above
marginal likelihood-to-evidence ratio estimation in parallel by setting
:math:`\boldsymbol{\vartheta}= (\theta_i)` for :math:`i=1, \dots, d`. To
this end, we utilize the factorized constrained prior,
:math:`\tilde{p}_r(\theta) = \tilde{p}_r(\theta_1)\cdots\tilde{p}_r(\theta_d)`,
which is defined recursively by a cutoff criterion,

.. math::

   \tilde{p}_{r}(\theta_{i})
       \propto
       p(\theta_{i}) \Theta_{H} \left[ \frac{\tilde{r}_{r-1}(\theta_{i}, \mathbf{x})}{\max_{\theta_{i}} \tilde{r}_{r-1}(\theta_{i}, \mathbf{x})} - \epsilon \right],
       \label{eqn:it}

where :math:`\Theta_{H}` denotes the Heaviside step function and
:math:`\epsilon` denotes the minimum likelihood-ratio which passes
through the threshold. We use
:math:`\tilde{p}_1(\boldsymbol{\theta}) = p(\boldsymbol{\theta})` as an
initial prior in the iterative scheme.

In every round, each 1-dim posterior approximates a marginalization of
the same underlying constrained posterior, allowing us to effectively
reuse training data and train efficiently in a multi-target regime. The
inference network is therefore divided into a featurizer
:math:`\mathbf{F}(\mathbf{x})` with shared parameters and a set of
:math:`d` independent Multi-layer Perceptons
:math:`\{\textrm{MLP}_i(\cdot, \cdot)\}_{i=1}^{d}` which estimate
individual 1-dim marginal posteriors and do not share parameters, such
that
:math:`f_{\phi}(\mathbf{x}, \theta_i) = \textrm{MLP}_i(\mathbf{F}(\mathbf{x}), \theta_i)`.

This technique is valid as long as the excluded prior regions do not
significantly affect the integrals in Eq. `[eqn:post] <#eqn:post>`__.
For uncorrelated parameters, a sufficient criterion is that the impact
on the marginal posteriors is small, which we guarantee through the
iteration criterion Eq. `[eqn:it] <#eqn:it>`__. In the case of a very
large number of strongly correlated parameters the algorithm can
inadvertently cut away tails of the marginal posteriors. Decreasing
:math:`\epsilon` mitigates this effect. Discussion is left for future
study :raw-latex:`\cite{swyft_future}`.

With this design, the posteriors from the final round can be used to
approximate the true 1-dim marginal posteriors,
:math:`\tilde{p}_{R}(\theta_i \vert \mathbf{x}_{0}) \approx p(\theta_i\vert \mathbf{x}_{0})`,
while previous rounds were used to iteratively focus on relevant parts
of the parameter space. The key result and value of NRE lies in the
utility of our constrained prior from round :math:`R`. The final
constrainted prior, along with previously generated and cached samples,
allows for estimation of *any* higher dimensional marginal posterior
:math:`\tilde{p}_R(\boldsymbol{\vartheta}|\mathbf{x}_0) \approx p(\boldsymbol{\vartheta}|\mathbf{x}_0)`
of interest by doing likelihood-to-evidence ratio estimation, often
without further simulation.

Inhomogeneous Poisson Point Process (iP3) Sample Caching.
=========================================================

Simulating
:math:`(\mathbf{x}, \boldsymbol{\theta})\sim p(\mathbf{x}|\boldsymbol{\theta})p(\boldsymbol{\theta})`
can be extremely expensive. We develop a scheme to systematically reuse
appropriate subsets of previous simulator runs. Our method samples
:math:`N\sim \text{Pois}(\hat N)` parameter vectors from an arbitrary
distribution :math:`p(\boldsymbol{\theta})`, where :math:`\hat N` is the
expected number of samples. Taking :math:`N` samples from
:math:`p(\boldsymbol{\theta})` is equivalent to drawing a single sample
:math:`\Theta \equiv \{\boldsymbol{\theta}^{(n)}\}_{n=1}^{N}` from an
inhomogenous Poisson point process (PPP) with intensity function
:math:`\lambda_{r}(\boldsymbol{\theta}) = \hat{N} p(\boldsymbol{\theta})`.
In this context, :math:`\Theta` is known as a set of *points*. This
formulation provides convenient mathematical properties
:raw-latex:`\cite{ppp}`, at the low price of introducing variance in the
number of samples drawn. The precise number of samples doesn’t matter as
long as :math:`N \approx \hat{N}`, which is true in our regime of order
:math:`\geq 1000`.

We will need two properties of PPPs. *Superposition:* Given two
independent PPPs with intensity functions
:math:`\lambda_{1}(\boldsymbol{\theta})` and
:math:`\lambda_{2}(\boldsymbol{\theta})`, the sum yields another PPP
with intensity function
:math:`\lambda(\boldsymbol{\theta}) = \lambda_{1}(\boldsymbol{\theta}) + \lambda_{2}(\boldsymbol{\theta})`.
The union of two sets of points :math:`\Theta = \Theta_1 \cup \Theta_2`
from the individual PPPs is equivalent to a single set of points from
the combined PPP. *Thinning:* Consider a PPP with intensity function
:math:`\lambda(\boldsymbol{\theta})`, and an arbitrary function
:math:`q(\boldsymbol{\theta}): \mathbb{R}^{d} \to [0, 1]`. If we are
interested in drawing from a PPP with intensity function
:math:`\lambda_{q}(\boldsymbol{\theta}) = q(\boldsymbol{\theta}) \lambda(\boldsymbol{\theta})`,
we can achieve this by drawing a set of points :math:`\Theta`
distributed like :math:`\lambda(\boldsymbol{\theta})` and then rejecting
individual points :math:`\boldsymbol{\theta}^{(n)}` with probability
:math:`1 - q(\boldsymbol{\theta}^{(n)})`.

The parameter cache is defined by a set of points :math:`\Theta_{sc}`
drawn from a PPP with intensity function
:math:`\lambda_{sc}(\boldsymbol{\theta})`. For every point
:math:`\boldsymbol{\theta}\in\Theta_{sc}`, a corresponding observation
:math:`\mathbf{x}` is stored in an observation cache
:math:`\mathcal{X}_{sc}`. The iP3 cache sampling algorithm that is
responsible for maintaining the caches and sampling from a PPP with
target intensity function
:math:`\lambda_t(\boldsymbol{\theta}) = \hat{N} p(\boldsymbol{\theta})`
is written out in the supplementary material. It is summarized in two
steps: First, consider all points
:math:`\boldsymbol{\theta}\in \Theta_{sc}` from the cache and accept
them with probability
:math:`\min(1, \lambda_t(\boldsymbol{\theta})/\lambda_{sc}(\boldsymbol{\theta}))`.
The thinning operation yields a sample :math:`\Theta_1` from a PPP with
intensity function
:math:`\lambda_1(\boldsymbol{\theta}) = \min(\lambda_t(\boldsymbol{\theta}), \lambda_{sc}(\boldsymbol{\theta}))`.
Second, draw a new set of points :math:`\Theta_p` from
:math:`\lambda_t(\boldsymbol{\theta})`, and accept each
:math:`\boldsymbol{\theta}\in\Theta_p` with probability
:math:`\max(0, 1-\lambda_{sc}(\boldsymbol{\theta})/\lambda_t(\boldsymbol{\theta}))`.
This yields a sample :math:`\Theta_2` from a PPP with intensity function
:math:`\lambda_2(\boldsymbol{\theta}) = \max(0, \lambda_t(\boldsymbol{\theta}) - \lambda_{sc}(\boldsymbol{\theta}))`.
Thanks to superposition, the union
:math:`\Theta_1 \cup \Theta_2 = \Theta_t` yields a sample from the PPP
with intensity function :math:`\lambda_t(\boldsymbol{\theta})`–the
sample we were looking for. We only need to run simulations on points
from :math:`\Theta_1`. Points in :math:`\Theta_2` already have
corresponding observations in :math:`\mathcal{X}_{sc}` which we can
reuse. Finally, the new parameters are appended to the set of points in
the parameter cache, :math:`\Theta_{sc} \to \Theta_{sc} \cup \Theta_2`.
Similar for :math:`\mathcal{X}_{sc}`. On the basis of the superposition
principle, the intensity function of the :math:`\Theta_{sc}` cache is
updated
:math:`\lambda_{sc}(\boldsymbol{\theta}) \to \max(\lambda_{sc}(\boldsymbol{\theta}), \lambda_t(\boldsymbol{\theta}))`.

Storing and updating the parameter cache’s intensity function
:math:`\lambda_{sc}(\boldsymbol{\theta})` can pose challenges when it is
complex and high-dimensional. Our NRE implementation overcomes these
challenges by learning marginal 1-dim posteriors, guaranteeing that the
relevant target intensities always factorize,
:math:`\lambda_t(\boldsymbol{\theta}) = \lambda_t(\theta_1)\cdots \lambda_t(\theta_d)`.
Storage of and calculation with factorizable functions simplifies
matters.
