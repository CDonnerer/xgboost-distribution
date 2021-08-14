=========
Changelog
=========


Current version
===============

Version 0.2.0, 2021-08-14
--------------------------

- Performed experiments on various datasets to quantify XGBDistribution performance
  - Added comparisons with NGBRegressor and XGBRegressor models
  - Self-contained script for experiment run
  - Detailed writeup in documentation of results
- Added exponential distribution
- Added Laplace distribution

Older versions
===============

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
