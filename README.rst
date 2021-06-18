.. image:: https://github.com/CDonnerer/xgb-dist/actions/workflows/test.yml/badge.svg?branch=main
  :target: https://github.com/CDonnerer/xgb-dist/actions/workflows/test.yml

.. image:: https://coveralls.io/repos/github/CDonnerer/xgb-dist/badge.svg?branch=main
  :target: https://coveralls.io/github/CDonnerer/xgb-dist?branch=main

.. image:: https://readthedocs.org/projects/xgb-dist/badge/?version=latest
  :target: https://xgb-dist.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status


============
xgb-dist
============

XGBoost for probabilistic prediction. Like `NGBoost`_, but faster and in the `XGBoost scikit-learn API`_.

.. image:: https://raw.githubusercontent.com/CDonnerer/xgb-dist/main/imgs/xgb_dist.png
    :align: center
    :width: 600px
    :alt: XGBDistribution example


Usage
===========

``XGBDistribution`` follows the `XGBoost scikit-learn API`_, except for an additional
keyword in the constructor for specifying the distribution. Given some data,
we can fit a model:

.. code-block:: python

      from sklearn.datasets import load_boston
      from sklearn.model_selection import train_test_split

      from xgb_dist import XGBDistribution

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

After fitting, we can predict the parameters of the distribution for new data.
This will return a namedtuple of numpy arrays for each parameter of the
distribution:

.. code-block:: python

      preds = model.predict(X_test)
      mean, scale = preds.mean, preds.scale


NGBoost performance comparison
===============================

``XGBDistribution`` follows the method shown in the `NGBoost`_ library, namely using
natural gradients to estimate the parameters of the distribution.

Below, we show a performance comparison of the `NGBoost`_ ``NGBRegressor`` and
``XGBDistribution`` models, using the Boston Housing dataset and a normal
distribution (similar hyperparameters). We note that while the performance of
the two models is essentially identical, XGBDistribution is **50x faster**
(timed on both fit and predict steps).

.. image:: https://raw.githubusercontent.com/CDonnerer/xgb-dist/main/imgs/performance_comparison.png
          :align: center
          :width: 600px
          :alt: XGBDistribution vs NGBoost


Full XGBoost features
======================

``XGBDistribution`` offers the full set of XGBoost features available in the
`XGBoost scikit-learn API`_, allowing, for example, probabilistic prediction with
`monotonic constraints`_:

.. image:: https://raw.githubusercontent.com/CDonnerer/xgb-dist/main/imgs/monotone_constraint.png
          :align: center
          :width: 600px
          :alt: XGBDistribution monotonic constraints


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.0.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.


.. _ngboost: https://github.com/stanfordmlgroup/ngboost
.. _xgboost scikit-learn api: https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
.. _monotonic constraints: https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html
