.. image:: https://github.com/CDonnerer/xgboost-distribution/actions/workflows/test.yml/badge.svg?branch=main
  :target: https://github.com/CDonnerer/xgboost-distribution/actions/workflows/test.yml

.. image:: https://coveralls.io/repos/github/CDonnerer/xgboost-distribution/badge.svg?branch=main
  :target: https://coveralls.io/github/CDonnerer/xgboost-distribution?branch=main

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black

.. image:: https://readthedocs.org/projects/xgboost-distribution/badge/?version=latest
  :target: https://xgboost-distribution.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/xgboost-distribution.svg
  :alt: PyPI-Server
  :target: https://pypi.org/project/xgboost-distribution/


====================
xgboost-distribution
====================

XGBoost for probabilistic prediction. Like `NGBoost`_, but `faster`_, and in the `XGBoost scikit-learn API`_.

.. image:: https://raw.githubusercontent.com/CDonnerer/xgboost-distribution/main/imgs/xgb_dist.png
    :align: center
    :width: 600px
    :alt: XGBDistribution example


Installation
============

.. code-block:: console

    $ pip install xgboost-distribution


`Dependencies`_:

.. code-block::

    python_requires = >=3.8

    install_requires =
        scikit-learn
        xgboost>=1.7.0


Usage
===========

``XGBDistribution`` follows the `XGBoost scikit-learn API`_, with an additional keyword
argument specifying the distribution, which is fit via `Maximum Likelihood Estimation`_:


.. code-block:: python

      from sklearn.datasets import fetch_california_housing
      from sklearn.model_selection import train_test_split

      from xgboost_distribution import XGBDistribution


      data = fetch_california_housing()
      X, y = data.data, data.target
      X_train, X_test, y_train, y_test = train_test_split(X, y)

      model = XGBDistribution(
          distribution="normal",
          n_estimators=500,
          early_stopping_rounds=10
      )
      model.fit(X_train, y_train, eval_set=[(X_test, y_test)])


See the `documentation`_ for all available distributions.

After fitting, we can predict the parameters of the distribution:

.. code-block:: python

      preds = model.predict(X_test)
      mean, std = preds.loc, preds.scale


Note that this returned a `namedtuple`_ of `numpy arrays`_ for each parameter of the
distribution (we use the `scipy stats`_ naming conventions for the parameters, see e.g.
`scipy.stats.norm`_ for the normal distribution).


NGBoost performance comparison
===============================

``XGBDistribution`` follows the method shown in the `NGBoost`_ library, using natural
gradients to estimate the parameters of the distribution.

Below, we show a performance comparison of ``XGBDistribution`` and the `NGBoost`_
``NGBRegressor``, using the Boston Housing dataset, estimating normal distributions.
While the performance of the two models is essentially identical (measured on negative
log-likelihood of a normal distribution and the RMSE), ``XGBDistribution`` is
**30x faster** (timed on both fit and predict steps):

.. image:: https://raw.githubusercontent.com/CDonnerer/xgboost-distribution/main/imgs/performance_comparison.png
          :align: center
          :width: 600px
          :alt: XGBDistribution vs NGBoost


Please see the `experiments page`_ for results across various datasets.


Full XGBoost features
======================

``XGBDistribution`` offers the full set of XGBoost features available in the
`XGBoost scikit-learn API`_, allowing, for example, probabilistic regression
with `monotonic constraints`_:

.. image:: https://raw.githubusercontent.com/CDonnerer/xgboost-distribution/main/imgs/monotone_constraint.png
          :align: center
          :width: 600px
          :alt: XGBDistribution monotonic constraints


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
.. _faster:  https://xgboost-distribution.readthedocs.io/en/latest/experiments.html
.. _xgboost scikit-learn api: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
.. _dependencies: https://github.com/CDonnerer/xgboost-distribution/blob/feature/update-linting/setup.cfg#L37
.. _monotonic constraints: https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html
.. _scipy.stats.norm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
.. _LAPACK gesv: https://www.netlib.org/lapack/lug/node71.html
.. _xgboost: https://github.com/dmlc/xgboost
.. _documentation: https://xgboost-distribution.readthedocs.io/en/latest/api/xgboost_distribution.XGBDistribution.html#xgboost_distribution.XGBDistribution
.. _experiments page: https://xgboost-distribution.readthedocs.io/en/latest/experiments.html
.. _numpy arrays: https://numpy.org/doc/stable/reference/generated/numpy.array.html
.. _scipy stats: https://docs.scipy.org/doc/scipy/reference/stats.html
.. _namedtuple: https://docs.python.org/3/library/collections.html#collections.namedtuple
.. _maximum likelihood estimation: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
