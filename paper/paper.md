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
 - name: Gravitation Astroparticle Physics Amsterdam (GRAPPA), University of Amsterdam, Science Park 904, 1098 XH Amsterdam
   index: 1
 - name: Amsterdam Machine Learning Lab (AMLab), University of Amsterdam, Science Park 904, 1098 XH Amsterdam
   index: 2
 - name: AI4Science Lab, University of Amsterdam, Science Park 904, 1098 XH Amsterdam
   index: 3
 - name: Netherlands eScience Center, Science Park 140, 1098 XG Amsterdam, The Netherlands
   index: 4
date: 12 December 2021
bibliography: paper.bib
---
# Summary
Parametric stochastic numerical simulators are ubiquitous in science. They model observed phenomena by mapping a parametric representation of simulation conditions to a hypothetical observation--effectively sampling from a probability distribution over observational data known as the likelihood. Simulators are advantageous because they easily encode relevant scientific knowledge. *Simulation-based inference* (SBI) is a machine learning technique which applies a simulator, a fitted statistical surrogate model, and a set of prior beliefs to estimate a probabilistic description of the parameters which plausibly generated some observational data. This description of parameters is known as the posterior and it is the end-product of Bayesian inference.


Our package `swyft` implements a specific, simulation-efficient SBI method called *Truncated Marginal Neural Ratio Estimation* (TMNRE) [@miller2021truncated]; it estimates the likelihood-to-evidence ratio to approximate the posterior, as in @Hermans2019. `swyft` [@swyft] provides a collection of tools to simulate and store data, locally or in a distributed computing setting, and perform (marginalized) simulation-based Bayesian inference. It produces ready-to-publish plots that demonstrate the calibration of the posterior estimate along with the posterior itself.


# Statement of Need
Estimating the posterior can be prohibitively expensive for complex data and slow simulators. Part of the reason is the sequential nature of likelihood-based Markov chain Monte-Carlo [@metropolis; @hastings]. In contrast, SBI parallelizes simulation in most circumstances, thereby reducing the practical waiting time for results. In pursuit of further simulation efficiency, @miller2021truncated argue that fitting the joint posterior for all parameters is unnecessary when a marginal estimate of the posterior will suffice. Some SBI methods are amortized, whereby the statistical model is fit to estimate posteriors for all possible observations simultaneously. While amortization enables necessary posterior calibration checks, like expected coverage probability [@miller2021truncated; @hermans2021averting], it is more efficient to fit the model on only a subset of the parameters that could have plausibly generated the observation.

`swyft` satisfies necessary requirements, like estimating the marginal posteriors of interest and enabling posterior calibration checks, while taking a lean approach to avoid all unnecessary simulation. In this pursuit, `swyft` truncates the prior to regions relevant for given observational data and reuses compatible existing simulations.  `swyft` automates irksome matters like distributed computing and data storage with `dask` [@dask] and `zarr` [@zarr] respectively. `swyft` is designed to:

1. Estimate arbitrary marginal posteriors, i.e., the posterior over parameters of interest, marginalizing over nuisance parameters.
2. Perform targeted inference by truncating the prior distribution with an indicator function estimated in a sequence of inferences.
3. Estimate the expected coverage probability of fully amortized SBI posteriors and locally amortized posteriors that are limited to truncated regions.
4. Seamlessly reuse simulations from previous analyses by drawing already-simulated data first via a flexible storage solution.
5. Integrate advanced distribution and storage tools to simplify application of complex simulators.

Although there is a rich ecosystem of SBI implementations, TMNRE did not naturally fit in an existing framework since it requires parallel estimation of marginal posteriors and a truncated prior. `swyft` does the parallel training of the ratio using another dimension in a PyTorch tensor and created a custom truncated prior data structure to overcome these challenges. `swyft` aims to meet the ever-increasing demand for efficient and testable Bayesian inference in fields like physics, cosmology, and astronomy by implementing TMNRE together with practical distributed computing and storage tools.

## Existing research with `swyft`
The software package has enabled inference on dark matter substructure in strongly lensed galaxies [@coogan2020targeted], estimated cosmological parameters from cosmic microwave background simulation data [@cole2021fast], and was cited in a white paper laying out a vision for astropartical physics research during the next decade [@batista2021eucapt]. Ongoing work with `swyft` aims to reduce the response time to gravitational wave triggers from LIGO-Virgo by estimating the marginal posterior with unprecedented speed. There is an existing proof-of-concept by @delaunoy2020lightning although the `swyft` software package was not applied. Generally, speeding up gravitational wave inference using simulation-based inference is an active area of research [@gabbard2022bayesian; @dax2021real; @chua2020learning]. In another work-in-progress, `swyft` helps to characterize the magnetohydrodynamics of binary neutron star mergers using multi-messenger gravitational and electrodynamic data where marginalization would be impossible with likelihood-based methods.

## Related theoretical work
There is a long tradition of likelihood-free inference, also known as *Approximate Bayesian Computation* (ABC), going back to as early as the 1980s [@diggle1984monte; @first_abc; @second_abc; @Toni2009-fd; @Beaumont2009-gl]. Traditional techniques use Monte-Carlo rejection sampling and are summarized by @sisson2018handbook and @karabatsos2018approximate. We track the development of classifiers for the estimation of likelihood ratios to a few references. @Cranmer2015 compared the ratio between the likelihood of a freely varying parameter and a fixed reference value for frequentist inference. @pham2014note estimated the ratio between likelihoods for Markov chain Monte-Carlo sampling. @thomas2016likelihood and @gutmann2018likelihood introduced the framework which allows for likelihood-to-evidence ratio estimation. Like `swyft`, @blum2010non proposed to truncate the prior for sampling but do so within an ABC scheme.

Modern SBI is a quickly evolving field that has several techniques under development [@Cranmer2020]. Neural network-based methods are categorized according to the term they approximate in Bayes' formula. `swyft` is a method which approximates the likelihood-to-evidence ratio $\frac{p(x \mid \theta)}{p(x)}$ where $\theta$ are the parameters and $x$ is the observational data. Works by @Hermans2019, @Durkan2020, and @rozet2021arbitrary are closely related to `swyft` as they also approximate the likelihood-to-evidence ratio. Like `swyft`, @rozet2021arbitrary estimate marginal posteriors, but unlike swyft, they attempt to amortize over all possible marginals with a single neural network. Other methods estimate the posterior directly [@epsilon_free; @lueckmann2017flexible; @greenberg2019automatic; @Durkan2020] or the likelihood itself [@papamakarios2019sequential; @lueckmann2019likelihood].

## Related software
`swyft` is unique because it implements TMNRE and a method for simulation reuse. It also offers sophisticated distributed simulation and storage tools coupled directly to the software. We briefly discuss the alternatives in the thriving ecosystem of SBI software packages.

`sbi` [@sbi] features a selection of modern neural SBI algorithms. It is accompanied by a benchmark `sbibm` [@sbibm] which tests those methods against a set of tractable toy problems. `pydelfi` [@pydelfi-repo] estimates the likelihood of a learned summary statistic [@alsing2018massive; @alsing2019fast]--`swyft` users should pay special attention to this repository since it can also project out nuisance parameters [@alsing2019nuisance]. `carl` [@louppe2016] uses a classifier to estimate the likelihood ratio as @Cranmer2015 did and `hypothesis` [@hypothesis-repo] includes several toy simulators.

Non-neural implementations for SBI also exist. `elfi` [@elfi2018] implements BOLFI, an algorithm based on Gaussian processes [@gutmann2016bayesian]. `pyabc` [@Klinger2018] and `ABCpy` [@dutta2017] are two suites of ABC algorithms.


# Description of software
`swyft` implements *Marginal Neural Ratio Estimation* (MNRE), a method which trains an amortized likelihood-to-evidence ratio estimator for any marginal posterior of interest. `swyft` makes it easy to estimate a set of marginals in parallel, e.g., for a corner plot. To use `swyft`, the operator must provide a quantification of prior beliefs, a python-callable or bash-scriptable simulator, and an observation-of-interest.

Performing TMNRE with `swyft`, by restricting simulation to a truncated prior region, is simple and demonstrated in the documentation. Constructing these truncated regions can be done manually or based on a previous inference. Routines are provided for all necessary plots and for calculating the expected coverage probability of a given likelihood-to-evidence ratio estimator. This calculation is essential as a sanity check to determine whether the approximate posterior is calibrated.

The machine learning aspects of `swyft` are implemented in PyTorch [@pytorch] while the truncated prior is implemented within `numpy` [@harris2020array]. Storing previously simulated data for reuse in later analyses is accomplished with `zarr` [@zarr] and parallelization of simulation is achieved with `dask` [@dask]. `swyft` has other important dependencies, namely `scipy` [@2020SciPy-NMeth], `seaborn` [@Waskom2021], `matplotlib` [@Hunter:2007], `pandas` [@reback2020pandas; @mckinney-proc-scipy-2010], and `jupyter` [@jupyter].

# Acknowledgements
The developers of `swyft` want to thank early adopters of the software Adam Coogan, Noemi Anau Montel, Kosio Karchev, Elias Dubbeldam, Uddipta Bhardwaj, and Ioannis Bousdoukos. We are grateful for the additional expertise offered by Patrick Forré and Samaya Nissanke.

Benjamin Kurt Miller is funded by the University of Amsterdam Faculty of Science (FNWI), Informatics Institute (IvI), and the Institute of Physics (IoP). This project received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 864035 -- UnDark). This work was also supported by the Netherlands eScience Center and SURF under grant number ETEC.2019.018. This work was carried out on the Dutch national e-infrastructure with the support of SURF Cooperative. We also would like to thank SURF for providing computational resources via the EINF-1194 grant.


# References
