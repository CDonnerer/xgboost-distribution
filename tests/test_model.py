"""Test suite for XGBDistribution model

To add:
- Pandas test

"""

import os
import pickle

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


def test_predict_before_fit_fails(small_X_y_data):
    X, y = small_X_y_data
    model = XGBDistribution(n_estimators=10)

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_get_base_margin():
    """Test that base_margin are created as expected for each sample"""
    model = XGBDistribution(distribution="normal")
    X = np.array([[1], [1]])
    y = np.array([1, 0])
    model.fit(X, y)

    margin = model._get_base_margin(n_samples=2)
    expected_margin = np.array([0.5, np.log(np.std(y)), 0.5, np.log(np.std(y))])
    np.testing.assert_array_equal(margin, expected_margin)


# -------------------------------------------------------------------------------------
#  Model IO tests
# -------------------------------------------------------------------------------------


def assert_model_equivalence(model_a, model_b, X):
    np.testing.assert_almost_equal(
        model_a._starting_params, model_b._starting_params, decimal=9
    )
    assert model_a._distribution.__class__ == model_b._distribution.__class__

    preds_a = model_a.predict(X)
    preds_b = model_b.predict(X)

    for param_a, param_b in zip(preds_a, preds_b):
        np.testing.assert_array_equal(param_a, param_b)


@pytest.mark.parametrize(
    "model_format",
    ["bst", "json"],
)
def test_XGBDistribution_save_and_load_model(small_X_y_data, model_format, tmpdir):
    X, y = small_X_y_data
    model = XGBDistribution(n_estimators=10)
    model.fit(X, y)

    model_path = os.path.join(tmpdir, f"model.{model_format}")
    model.save_model(model_path)

    saved_model = XGBDistribution()
    saved_model.load_model(model_path)

    assert_model_equivalence(model_a=model, model_b=saved_model, X=X)


def test_XGBDistribution_pickle_dump_and_load(small_X_y_data, tmpdir):
    X, y = small_X_y_data

    model = XGBDistribution(n_estimators=10)
    model.fit(X, y)

    model_path = os.path.join(tmpdir, "model.pkl")

    with open(model_path, "wb") as fd:
        pickle.dump(model, fd)

    with open(model_path, "rb") as fd:
        saved_model = pickle.load(fd)

    assert_model_equivalence(model_a=model, model_b=saved_model, X=X)
