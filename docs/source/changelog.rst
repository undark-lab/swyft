Changelog
=========

The full release logs can be found on `github
<https://github.com/undark-lab/swyft/releases>_.  Here we provide a brief
summary.

v0.4.4 (2023-07-31)
-------------------

- Included updated author list (AUTHOR file) and update LICENSE information
- Add torch conjugate gradient to dependencies (for forthcoming applications)
- Add list of Swyft publications and events to docs
- pytorch-lighthing 2.0 compatibility not fully established (pending until 2.0 has feature parity with 1.9)
- Add swyft.get_class_probs for extracting classification results from ratio estimators
- Add LogRatioEstimator_Gaussian, a Gaussian ratio estimator for arbitrary dimensions
- Deprecate LogRatioEstimator_1dim_Gaussian (worked only with 1dim features and 1dim parameters)
- Rewritten logratios aggregator in swyft.Module to be more resilient 
- Add AuxLoss, which just enables to add arbitrary extra losses (e.g. related to regularisation) to the training object
- Enable Spectral embedding for parameters in LogRatioEstimator_Ndim for high fidelity posteriors
- Add LogRatioEstimator_Autoregressive, an autoregressive neural ratio estimator module

v0.4.3 (2023-04-28)
-------------------

- Tutorials completely redone (they are the ones from the Jan 2023 training event)
- New documentation theme, and general documentation clean-up
- pytorch-lightning 2.0.0 compatibility


v0.4.2 (2023-04-26)
-------------------

Various minor bug fixes and interface improvements. 


v0.4.1 (2022-11-12)
-------------------

Bugfix release.


v0.4.0 (2022-11-09)
-------------------

Release of Swyft with new pytorch-lightning API.


v0.4.0-pre2 (2022-10-04)
------------------------

All API components are now in a reasonably stable state, code has been
reorganized and docstrings updated.  Next steps will include polishing
docstrings and generalizing some of the functionality, but should leave the
central API components untouched.  Swyft has a logo now.


v0.4.0-pre1 (2022-08-25)
------------------------

Initial alpha release of Swyft based on pytorch-lightning.
