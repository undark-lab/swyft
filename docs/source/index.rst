.. swyft documentation master file, created by
   sphinx-quickstart on Thu Nov 12 11:23:08 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. warning::
   SWYFT is research software under heavy development and still in its alpha
   phase.  There are many rough edges, and things might break.  However, the
   core algorithms work, and we use SWYFT in production for research papers.
   If you encounter problems, please contact the authors or submit a bug
   report.**

Welcome to swyft's documentation!
=================================

*Cursed by the dimensionality of your nuisance space? Wasted by Markov
chains that reject your simulations? Exhausted from messing with
simplistic models, because your inference algorithm cannot handle the
truth? Try SWYFT for some pain relief.*

**SWYFT aims at combining the convenience and precision of traditional
likelihood-based inference with the flexibility and power of modern neural
network based likelihood-free methods.  SWYFT is based on the concepts of
Amortized Approximate Likelihood Ratio Estimation (AALR) [1], Nested Ratio
Estimation (NRE, [2]) and the iP3 sample caching ([2]). In short, SWYFT = AALR
+ NRE + iP3.**



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   goals
   context
   quickstart
   theorytex

Autodoc documentation
=====================

.. automodule:: swyft
  :members:
  :undoc-members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
