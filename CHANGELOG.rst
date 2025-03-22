=========
Changelog
=========

Development version
===================

Version 0.4.x, 20xx-xx-xx
--------------------------

- ..


Current version
===============

Version 0.4.0, 2025-03-22
--------------------------

- Fix for xgboost>=3.0.0 compatibility (requires python >=3.10)
- Use `xgboost-cpu` where possible to reduce package footprint


Older versions
===============


Version 0.3.1, 2025-02-15
--------------------------

- Fix for sklearn >= 1.6 compatibility, :issue:`104`


Version 0.3.0, 2024-07-06
--------------------------

- Adapted for xgboost==2.1.0 compatibility
- Numpy compatibility for beyond 2.0.0 (older versions are still supported)


Version 0.2.9, 2023-12-10
--------------------------

- Fix float32 serialisation, :issue:`97`


Version 0.2.8, 2023-09-24
--------------------------

- Fixes to make package work with xgboost 2.0
- Various internal changes for xgboost sklearn API consistency


Version 0.2.7, 2023-03-12
--------------------------

- Fix overflow issues for normal distribution, :issue:`64`
- Removed verbosity hack in model training
- Better support for pickle/joblib, :issue:`82`


Version 0.2.6, 2023-01-21
--------------------------

- Added support for sample weights, :issue:`45`


Version 0.2.5, 2022-11-01
--------------------------

- Added example script for hyperparameter tuning
- Python requires >= 3.8 & xgboost >= 1.7.0 compatibility


Version 0.2.4, 2022-04-23
--------------------------

- Added more precise loss description, negative log likelihood vs error
- Various updates to conform with xgboost==1.6.0 release


Version 0.2.3, 2021-12-28
--------------------------

- Added type hints to XGBDistribution model class
- Hotfix to add error raising if sample weights are used (which is not yet implemented)


Version 0.2.2, 2021-10-23
--------------------------

- Hot fix to enable compatibility with xgboost v1.5.0 (enable_categorical kwarg)


Version 0.2.1, 2021-10-10
--------------------------

- Fixed the objective parameter in trained model to be reflective of distribution
- Support for model saving and loading with pickle (please don't use pickle)
- Added count data example with distribution heatmap, :issue:`45`
- Updated docs to include estimators parameter, :issue:`43`
- Implemented cleaner model saving, tests against binary and json formats


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
