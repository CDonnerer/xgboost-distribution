=========
Changelog
=========

Development version
===================

Version 0.2.X, 2021-xx-xx
--------------------------

- ...


Current version
===============

Version 0.2.1, 2021-10-10
--------------------------

- Fixed the objective parameter in trained model to be reflective of distribution
- Support for model saving and loading with pickle (please don't use pickle)
- Added count data example with distribution heatmap, :issue:`45`
- Updated docs to include estimators parameter, :issue:`43`
- Implemented cleaner model saving, tests against binary and json formats


Older versions
===============

Version 0.2.0, 2021-08-14
--------------------------

- Performed experiments on various datasets to assess XGBDistribution performance
- Added exponential distribution
- Added Laplace distribution

Version 0.1.2, 2021-07-10
-------------------------

- Added poisson distribution
- Added negative-binomial distribution
- Changed naming conventions of distributions
- Safety checks on distribution parameters


Version 0.1.1, 2021-07-01
-------------------------

- Added lognormal distribution
- Cleanup of distribution code, tested
- Silenced warnings during fit and predict steps
- Explicit link to RTD, showing available distributions
- CI tests running in Python 3.6, 3.7, 3.8


Version 0.1.0, 2021-06-20
-------------------------

- First release of xgboost-distribution package
- Contains XGBDistribution estimator, an xgboost wrapper with natural gradients
- Normal distribution implemented
