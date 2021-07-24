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
additional keyword argument specifying the distribution (see the
`documentation`_ for a full list of available distributions):

.. code-block:: python

      from sklearn.datasets import load_boston
      from sklearn.model_selection import train_test_split

      from xgboost_distribution import XGBDistribution


      data = load_boston()
      X, y = data.data, data.target
      X_train, X_test, y_train, y_test = train_test_split(X, y)

      model = XGBDistribution(distribution="normal", n_estimators=500)
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
distribution. We note that while the performance of the two models is essentially
identical, XGBDistribution is **50x faster** (timed on both fit and predict steps):


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

Across various datasets, we find ``XGBDistribution`` **performs similarly**
to ``NGBRegressor``, but is typically at least an **order of magnitude faster**.
For example, for the largest dataset, MSD, we found that `XGBDistribution``
took an average of 18 minutes vs 6.7 hours for ``NGBRegressor``:

+----------------+---------------------------------------+-------------------------------------+---------------------------+
|                | XGBDistribution                       | NGBRegressor                        |  XGBRegressor             |
+---------+------+-----------+-----------+---------------+-----------+-----------+-------------+-----------+---------------+
| Dataset | N    | NLL       | RMSE      | Time  (s)     | NLL       | RMSE      | Time (s)    | RMSE      | Time (s)      |
+=========+======+===========+===========+===============+===========+===========+=============+===========+===============+
| Boston  |506   | 2.62(26)  | 3.41(69)  | 0.067(1)      | 2.55(24)  | 3.25(66)  | 2.68(45)    | 3.27(65)  | 0.035(1)      |
+---------+------+-----------+-----------+---------------+-----------+-----------+-------------+-----------+---------------+
| Concrete|1030  | 3.14(21)  | 5.41(74)  | 0.13(3)       | 3.09(13)  | 5.62(69)  | 5.79(59)    | 4.38(70)  | 0.09(2)       |
+---------+------+-----------+-----------+---------------+-----------+-----------+-------------+-----------+---------------+
| Energy  |768   | 0.58(41)  | 0.45(7)   | 0.15(3)       | 0.62(28)  | 0.49(7)   | 5.33(35)    | 0.40(6)   | 0.05(2)       |
+---------+------+-----------+-----------+---------------+-----------+-----------+-------------+-----------+---------------+
| Naval   |11934 | -5.11(6)  | 0.0014(1) | 5.8(8)        | -3.91(2)  | 0.0059(1) | 43.6(5)     | 0.00123(5)| 1.93(7)       |
+---------+------+-----------+-----------+---------------+-----------+-----------+-------------+-----------+---------------+
| Power   |9568  | 2.77(11)  | 3.79(24)  | 1.21(52)      | 2.77(7)   | 3.93(19)  | 14.9(3.1)   | 3.31(22)  | 0.59(19)      |
+---------+------+-----------+-----------+---------------+-----------+-----------+-------------+-----------+---------------+
| Protein |45730 | 2.81(4)   | 4.35(12)  | 12.2(4.0)     | 2.91(1)   | 4.78(5)   | 146.5(1.8)  | 4.09(7)   | 8.26(1.4)     |
+---------+------+-----------+-----------+---------------+-----------+-----------+-------------+-----------+---------------+
| Wine    |1588  | 0.98(15)  | 0.63(11)  | 0.11(3)       | 0.93(7)   | 0.62(3)   | 4.85(99)    | 0.62(3)   | 0.035(13)     |
+---------+------+-----------+-----------+---------------+-----------+-----------+-------------+-----------+---------------+
| Yacht   |308   | 0.89(1.1) | 0.76(29)  | 0.093(25)     | 0.75(64)  | 0.74(28)  | 4.95(50)    | 0.74(37)  | 0.047(35)     |
+---------+------+-----------+-----------+---------------+-----------+-----------+-------------+-----------+---------------+
| MSD     |515345| 3.450(4)  | 9.03(4)   | 18.9(1.5) (m) | 3.526(4)  | 9.74(4)   | 6.70(7) (h) | 8.97(3)   | 16.3(1.7) (m) |
+---------+------+-----------+-----------+---------------+-----------+-----------+-------------+-----------+---------------+



Note that for point estimates (RMSE), ``XGBRegressor`` offers the best performance.
Compared with ``XGBRegressor``, ``XGBDistribution`` will incur some performance
and speed penalty for providing a probabilistic regression.


Methodology
-------------------

We used 10-fold cross-validation, in each training fold 10% of the data were
used as a validation set for early stopping (repeated over 5 random seeds.)
The negative log-likelihood (NLL) and root mean squared error (RMSE) were estimated
for each test set, the above are the mean and standard deviation of these metrics
(across folds and random seeds).

All hyperparameters were defaults, except for ``max_depth=3`` in ``XGBDistribution``
and ``XGBRegressor``, since this is the default value of ``NGBRegressor``.
``XGBDistribution`` and ``NGBRegressor`` estimated normal distributions.

Please see the `benchmarking script`_ for full details.


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
.. _benchmarking script: https://github.com/CDonnerer/xgboost-distribution/blob/benchmarking/examples/benchmarking.py
