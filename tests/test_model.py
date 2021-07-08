import os

import pytest

import numpy as np
from sklearn.exceptions import NotFittedError

from xgboost_distribution.model import XGBDistribution


def test_XGBDistribution_early_stopping_fit(small_train_test_data):
    """Dummy test that everything runs"""

    X_train, X_test, y_train, y_test = small_train_test_data
    model = XGBDistribution(distribution="normal", max_depth=3, n_estimators=500)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)

    evals_result = model.evals_result()

    assert model.best_iteration == 6
    assert isinstance(evals_result, dict)


def test_XGBDistribution_early_stopping_predict(small_train_test_data):
    """Check that predict with early stopping uses correct ntrees"""
    X_train, X_test, y_train, y_test = small_train_test_data

    model = XGBDistribution(distribution="normal", max_depth=3, n_estimators=500)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)

    mean, var = model.predict(X_test)
    mean_iter, var_iter = model.predict(
        X_test, iteration_range=(0, model.best_iteration + 1)
    )
    np.testing.assert_array_equal(mean, mean_iter)
    np.testing.assert_array_equal(var, var_iter)


def test_distribution_set_param(small_X_y_data):
    """Check that updating the distribution params works"""

    X, y = small_X_y_data

    model = XGBDistribution(distribution="abnormal")
    model.set_params(**{"distribution": "normal"})

    model.fit(X, y)
    params = model.predict(X)

    assert isinstance(params[0], np.ndarray)


def test_XGBDistribution_save_and_load(small_X_y_data, tmpdir):
    X, y = small_X_y_data

    model = XGBDistribution(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)

    model_path = os.path.join(tmpdir, "model.bst")
    model.save_model(model_path)

    saved_model = XGBDistribution()
    saved_model.load_model(model_path)
    saved_preds = saved_model.predict(X)

    np.testing.assert_array_equal(preds[0], saved_preds[0])
    np.testing.assert_array_equal(preds[1], saved_preds[1])


def test_predict_before_fit_fails(small_X_y_data):
    X, y = small_X_y_data
    model = XGBDistribution(n_estimators=10)

    with pytest.raises(NotFittedError):
        model.predict(X)
