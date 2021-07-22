.. image:: https://github.com/CDonnerer/xgboost-distribution/actions/workflows/test.yml/badge.svg?branch=main
  :target: https://github.com/CDonnerer/xgboost-distribution/actions/workflows/test.yml

.. image:: https://coveralls.io/repos/github/CDonnerer/xgboost-distribution/badge.svg?branch=main
  :target: https://coveralls.io/github/CDonnerer/xgboost-distribution?branch=main

.. image:: https://readthedocs.org/projects/xgboost-distribution/badge/?version=latest
  :target: https://xgboost-distribution.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/xgboost-distribution.svg
  :alt: PyPI-Server
  :target: https://pypi.org/project/xgboost-distribution/


====================
xgboost-distribution
====================

XGBoost for probabilistic prediction. Like `NGBoost`_, but faster, and in the `XGBoost scikit-learn API`_.

.. image:: https://raw.githubusercontent.com/CDonnerer/xgboost-distribution/main/imgs/xgb_dist.png
    :align: center
    :width: 600px
    :alt: XGBDistribution example


Installation
============

.. code-block:: console

    $ pip install --upgrade xgboost-distribution


Usage
===========

``XGBDistribution`` follows the `XGBoost scikit-learn API`_, with an
additional keyword in the constructor for specifying the distribution (see the
`documentation`_ for a full list of available distributions):

.. code-block:: python

      from sklearn.datasets import load_boston
      from sklearn.model_selection import train_test_split

      from xgboost_distribution import XGBDistribution


      data = load_boston()
      X, y = data.data, data.target
      X_train, X_test, y_train, y_test = train_test_split(X, y)

      model = XGBDistribution(
          distribution="normal",
          n_estimators=500
      )
      model.fit(
          X_train, y_train,
          eval_set=[(X_test, y_test)],
          early_stopping_rounds=10
      )

After fitting, we can predict the parameters of the distribution:

.. code-block:: python

      preds = model.predict(X_test)
      mean, std = preds.loc, preds.scale


Note that this returned a namedtuple of numpy arrays for each parameter of
the distribution (we use the scipy naming conventions, see e.g. `scipy.stats.norm`_).


NGBoost performance comparison
===============================

``XGBDistribution`` follows the method shown in the `NGBoost`_ library, using
natural gradients to estimate the parameters of the distribution.

Below, we show a performance comparison of the `NGBoost`_ ``NGBRegressor`` and
``XGBDistribution`` models, using the Boston Housing dataset and a normal
distribution (similar hyperparameters). We note that while the performance of
the two models is essentially identical, XGBDistribution is **50x faster**
(timed on both fit and predict steps):


.. image:: https://raw.githubusercontent.com/CDonnerer/xgboost-distribution/main/imgs/performance_comparison.png
          :align: center
          :width: 600px
          :alt: XGBDistribution vs NGBoost


Please see below for detailed benchmarking results.

Full XGBoost features
======================

``XGBDistribution`` offers the full set of XGBoost features available in the
`XGBoost scikit-learn API`_, allowing, for example, probabilistic regression
with `monotonic constraints`_:

.. image:: https://raw.githubusercontent.com/CDonnerer/xgboost-distribution/main/imgs/monotone_constraint.png
          :align: center
          :width: 600px
          :alt: XGBDistribution monotonic constraints


Benchmarking
======================

Across a variety of datasets, ``XGBDistribution`` performs similarly to
``NGBRegressor``, but is substantially faster.

+--------------+------------------------------------+-----------------------------------+------------------------+
|              | XGBDistribution                    | NGBRegressor                      |  XGBRegressor          |
+---------+----+-----------+-----------+------------+-----------+-----------+-----------+-----------+------------+
| Dataset | N  | NLL       | RMSE      | Time (s)   | NLL       | RMSE      | Time (s)  | RMSE      | Time (s)   |
+=========+====+===========+===========+============+===========+===========+===========+===========+============+
| Boston  |506 | 2.62±0.26 | 3.41±0.69 | 0.067±0.01 | 2.55±0.24 | 3.25±0.66 | 2.68±0.45 | 3.27±0.65 | 0.035±0.01 |
+---------+----+-----------+-----------+------------+-----------+-----------+-----------+-----------+------------+
| Concrete|1030| 3.14±0.21 | 5.41±0.74 | 0.13±0.03  | 3.09±0.13 | 5.62±0.69 | 5.79±0.59 | 4.38±0.70 | 0.09±0.02  |
+---------+----+-----------+-----------+------------+-----------+-----------+-----------+-----------+------------+

We used 10-fold cross-validation, where in each training fold 10% of the data
were split off as a validation set, repeated over 5 random seeds. All models were
trained using early stopping on this validation set. The negative log-likelihood
(NLL) and root mean squared error (RMSE) were estimated for each test set, the
above quote the mean and standard deviation of these metrics.

All hyperparameters were defaults, except for ``max_depth=3`` in ``XGBDistribution``
and ``XGBRegressor``, since this is the default value of ``NGBRegressor``. Both
``XGBDistribution`` and ``NGBRegressor`` estimated normal distributions.


Acknowledgements
=================

This package would not exist without the excellent work from:

- `NGBoost`_ - Which demonstrated how gradient boosting with natural gradients
  can be used to estimate parameters of distributions. Much of the gradient
  calculations code were adapted from there.

- `XGBoost`_ - Which provides the gradient boosting algorithms used here, in
  particular the ``sklearn`` APIs were taken as a blue-print.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.0.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.


.. _ngboost: https://github.com/stanfordmlgroup/ngboost
.. _xgboost scikit-learn api: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
.. _monotonic constraints: https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html
.. _scipy.stats.norm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
.. _LAPACK gesv: https://www.netlib.org/lapack/lug/node71.html
.. _xgboost: https://github.com/dmlc/xgboost
.. _documentation: https://xgboost-distribution.readthedocs.io/en/latest/api/xgboost_distribution.XGBDistribution.html#xgboost_distribution.XGBDistribution
