Goals and plans
===============

Goals
-----

We are developing *swyft* to help solving our data analysis problems from
various fields of astroparticle physics, and make an effort that the software
will be useufl for others as well.  We have the following goals in mind.

- Convenience: SWYFT enables neural network based inference with the
  convenience of traditional sampling tools.
- Precision: SWYFT is as precise as likelihood-based methods where those can be
  applied, and flexible enough to cover the wide range of scenarios where they
  fail.
- Efficiency: SWYFT is simulator efficient, and automatically re-uses previous
  simulator runs in new analyses.

The current version of the code provides a prototype implementation of
algorithms that are capable of fullfilling the above goals.


Plans
-----

- Automatized parallelization of simulator runs on computing clusters using
  `dask`.
- Automatized detection of optimal network structures by hyper-parameter
  optimization on GPU clusters.
- Automatized consistency and coverage checks of estimated posteriors.
- Convenient handling of cached simulations, e.g. changes of priors and
  parameter ranges.
- Flexible handling of high-dimensional constrained priors.

.. note::
   Our ultimate goal is to develop *swyft* into a flexible and reliable
   inference tool, in collaboration with engineers from the Dutch eScience
   center and SURFsara, as part of the eTEC-BIG grant “Dark Generators”. Any
   feedback will be useful to shape the development of this tool. Stay tuned!


