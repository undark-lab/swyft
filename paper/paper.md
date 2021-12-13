---
title: 'swyft: Truncated Marginal Neural Ratio Estimation in Python'
tags:
  - Python
  - simulation-based inference
  - likelihood-free inference
  - machine learning
  - bayesian inference
  - system identification
  - parameter identification
  - inverse problem
authors:
  - name: Benjamin Kurt Miller
    orcid: 0000-0003-0387-8727
    affiliation: "1, 2, 3"
  - name: Alex Cole
    orcid: 0000-0001-8035-4308
    affiliation: 1
  - name: Christoph Weniger
    orcid: 0000-0001-7579-8684
    affiliation: 1
  - name: Francesco Nattino
    orcid: 0000-0003-3286-0139
    affiliation: 4
  - name: Ou Ku
    orcid: 0000-0002-5373-5209
    affiliation: 4
  - name: Meiert W. Grootes
    orcid: 0000-0002-5733-4795
    affiliation: 4
affiliations:
 - name: Gravitation Astroparticle Physics Amsterdam (GRAPPA), University of Amsterdam
   index: 1
 - name: Amsterdam Machine Learning Lab (AMLab), University of Amsterdam
   index: 2
 - name: AI4Science Lab, University of Amsterdam
   index: 3
 - name: Netherlands eScience Center
   index: 4
date: 12 December 2021
bibliography: paper.bib
---

# Summary
Parametric stochastic numerical simulators are ubiquitous in science. They model observed phenomena by mapping a parametric representation of simulation conditions to a hypothetical observation--effectively sampling from a complex probability distribution over observational data known as the likelihood. Simulators are advantageous because they easily encode relevant scientific knowledge. However, the simulation-based, implicit representation of the likelihood makes it difficult to find the set of parameters which plausibly generated some observational data--an incredibly common task called the inverse problem. Simulation-based inference is a machine learning technique which estimates a probabilistic description of this parameter set, known as the Bayesian posterior, by fitting a surrogate statistical model to parameters and simulated data.

Our package `swyft` implements a specific simulation-based inference method called Truncated Marginal Neural Ratio Estimation (TMNRE) [@miller2021truncated; @swyft]. `swyft` accomplishes four primary goals: a) Estimate arbitrary marginal posteriors, i.e., the posterior over parameters of interest, marginalizing over nuisance parameters, b) perform targeted inference by truncating the prior distribution--this combines simulation efficiency with empirical testability, c) seamlessly reuse simulations drawn from previous analyses, and d) integrate advanced distribution and storage tools to simplify application of complex simulators.

# Motivation
TODO (explain why this is necessary given all of that development with related work)

## Existing research with `swyft`
The existing software package has enabled inference on dark matter substructure in strongly lensed galaxies [@coogan2020targeted], has estimated cosmological parameters from cosmic microwave background simulation data [@cole2021fast], and was cited in a white paper laying out a vision for astropartical physics research during the next decade [@batista2021eucapt]. Ongoing work with `swyft` aims to reduce the response time to gravitational wave triggers from LIGO-Virgo by estimating the marginal posterior with unprecedented speed. In another project, `swyft` helps to characterize the magnetohydrodynamics of binary neutron star mergers using multi-messenger gravitational and electrodynamic data where marginalization would be impossible with likelihood-based methods.

## Related work
There is a long tradition of likelihood-free inference, also known as Approximate Bayesian Computation (ABC), going back to as early as the 1980s [@diggle1984monte; @first_abc; @second_abc; @Toni2009-fd; @Beaumont2009-gl]. Traditional techniques use Monte-Carlo rejection sampling and are summarized within @sisson2018handbook and @karabatsos2018approximate. We track the development of classifiers for the estimation of likelihood ratios to a few references. @Cranmer2015 compares the ratio between the likelihood of a freely varying parameter and a fixed reference value for frequentist inference. @pham2014note estimated the ratio between likelihoods for Markov chain Monte-Carlo sampling. @thomas2016likelihood and @gutmann2018likelihood introduced the framework which allows for likelihood-to-evidence ratio estimation. Like `swyft`, @blum2010non proposes to truncate the prior for sampling but it does so within an ABC scheme.

Modern simulation-based inference is a quickly evolving field that has several techniques under development [@Cranmer2020]. There have been extensive investigations on the suitability of simulation-based inference posteriors for science [@hermans2021averting]. The different methods are categorized by the term they approximate in Bayes' formula

$$p(\theta \mid x) = \frac{p(x \mid \theta)}{p(x)} p(\theta),$$

where $\theta$ are the parameters and $x$ is the observational data. The categories for neural approximation methods are given:

- Likelihood-to-evidence ratio estimation approximates $\frac{p(x \mid \theta)}{p(x)}$ and was developed in @Hermans2019 and @Durkan2020.
- Posterior estimation approximates $p(\theta \mid x)$. The relevant papers include @epsilon_free, @lueckmann2017flexible, @greenberg2019automatic, and @Durkan2020.
- Likelihood estimation approximates $p(x \mid \theta)$, as seen in @papamakarios2019sequential and @lueckmann2019likelihood.

There are a number of relevant and notable software repositories. `sbi` [@sbi] is a fully-featured software package that implements a selection of modern methods. It is accompanied by a benchmark `sbibm` [@sbibm] which tests those methods against a set of tractable toy problems. `pydelfi` [@pydelfi-repo] estimates the likelihood of a learned summary statistic [@alsing2018massive; @alsing2019fast]. It is particularly relevant for `swyft` users due to the work on projecting out nuisance parameters [@alsing2019nuisance]. `carl` [@louppe2016] uses a classifier to estimate the likelihood ratio as in @Cranmer2015 and `hypothesis` [@hypothesis-repo] includes several toy simulators.

Non-neural implementations for simulation-based inference also exist. `elfi` [@elfi2018] implements BOLFI, an algorithm based on Gaussian processes [@gutmann2016bayesian]. `pyabc` [@Klinger2018] and `ABCpy` [@dutta2017] are two suites of ABC algorithms.


# Description of software
TODO

The machine learning aspects of `swyft` are implemented in PyTorch [@pytorch] while the truncated prior is implemented within `numpy` [@harris2020array]. Storing previously simulated data for reuse in later analyses is acomplished with `zarr` [@zarr] and parallelization of simulation is achieved with `dask` [@dask]. `swyft` has other importance dependencies, namely `scipy` [@2020SciPy-NMeth], `seaborn` [@Waskom2021], `matplotlib` [@Hunter:2007], `pandas` [@reback2020pandas; @mckinney-proc-scipy-2010], and `jupyter` [@jupyter].

# Acknowledgements
TODO

This work was supported by the Netherlands eScience Center and SURF under grant number ETEC.2019.018. We also would like to thank SURF for providing computational resources via the EINF-1194 grant.


# References
