"""conftest.py for xgboost_distribution.
"""
import pytest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@pytest.fixture
def small_X_y_data():
    """Small set of X, y data (single feature)"""

    def true_function(X):
        return np.sin(3 * X)

    def true_noise_scale(X):
        return np.abs(np.cos(X))

    np.random.seed(1234)
    n_samples = 100
    X = np.random.uniform(-2, 2, n_samples)
    y = true_function(X) + np.random.normal(scale=true_noise_scale(X), size=n_samples)

    return X[..., np.newaxis], y


@pytest.fixture(params=["numpy", "pandas", "numpy-float32"])
def small_train_test_data(request, small_X_y_data):
    """Small set of train-test split X, y data (single feature)"""
    X, y = small_X_y_data

    if request.param == "pandas":
        X, y = pd.DataFrame(X), pd.Series(y)

    elif request.param == "numpy-float32":
        X, y = X.astype("float32"), y.astype("float32")

    return train_test_split(X, y, test_size=0.2, random_state=1)


@pytest.fixture
def small_X_y_count_data():
    """Small set of X, y data, with y being counts (positive int)"""

    def generate_count_data(n_samples=100):
        np.random.seed(11)  # 'tuned' to be simple to test against
        X = np.random.uniform(-2, 0, n_samples)
        n = 66 * np.abs(np.cos(X))
        p = 0.5 * np.abs(np.cos(X / 3))

        y = np.random.negative_binomial(n=n, p=p, size=n_samples)
        return X[..., np.newaxis], y

    return generate_count_data(n_samples=100)


@pytest.fixture(params=["numpy", "pandas"])
def small_train_test_count_data(request, small_X_y_count_data):
    """Small set of train-test split X, y data (positive int)"""
    X, y = small_X_y_count_data

    if request.param == "pandas":
        X, y = pd.DataFrame(X), pd.Series(y)

    return train_test_split(X, y, test_size=0.2, random_state=1)
