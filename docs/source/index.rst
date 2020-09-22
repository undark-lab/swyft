.. swyft documentation master file, created by
   sphinx-quickstart on Mon Aug 19 15:05:09 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to swyft's documentation!
========================================

SWYFT is a flexible and powerful tool for simulator efficient neural marginal posterior estimation.  It is meant as a drop-in replacement for conventional sampling based tools.  Although it is based on likelihood-free concepts, it reproduces results of likelihood-based analyses to high accuracy, usually with orders of magnitude less simulator runs.  At the same time, it is applicable to problems with large numbers of nuisance parameters, which are outside the scope of likelihood-based tools.  Lastly, in contrast to Markov-chain based approaches, simulator runs are never rejected, and automatically re-used in future analyses.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   theory


Autodoc documentation
=====================

.. automodule:: swyft.core
  :members:

.. automodule:: swyft.interface
  :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
