Goals and plans
===============


Overview
--------

*swyft* aims at combining the convenience and precision of traditional
likelihood-based inference with the flexibility and power of modern neural
network based likelihood-free methods.  *swyft* is based on the concepts of
Amortized Approximate Likelihood Ratio Estimation (AALR), Nested Ratio
Estimation (NRE), and iP3 sample caching. In short, *swyft* = AALR + NRE +
iP3.


Our goals
---------

We are developing *swyft* to help solve our data analysis problems in
astroparticle physics, and provide a useful software tool for others simultanously.
We have the following goals in mind.

- Convenience: *swyft* utilizes neural networks for likelihood-free inference with
  the simplicity of traditional sampling tools.
- Precision: *swyft* is as precise as likelihood-based methods, where those can be
  applied, and flexible enough to cover the wide range of scenarios where they
  fail.
- Efficiency: *swyft* is simulator efficient, and automatically re-uses previous
  simulator runs in new analyses.

The current version of the code provides a prototype implementation of
algorithms that are capable of fullfilling the above goals.


Future plans
------------

- Automatized parallelization of simulator runs on computing clusters using
  `dask`.
- Automatized detection of optimal network structures by hyper-parameter
  optimization on GPU clusters.
- Automatized consistency and coverage checks of estimated posteriors.
- Convenient handling of cached simulations, e.g. changes of priors and
  parameter ranges.
- Flexible handling of high-dimensional constrained priors.

.. note::
   The future development of *swyft* happens in collaboration with engineers
   from the Dutch `eScience center <https://www.esciencecenter.nl/>`_ and
   `SURFsara <https://surf.nl>`_, as part of the eTEC-BIG grant `Dark
   Generators <https://www.esciencecenter.nl/projects/darkgenerators/>`_. Any
   feedback will be useful to shape the development of this tool. Stay tuned!
