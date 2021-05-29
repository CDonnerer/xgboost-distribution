"""conftest.py for xgb_dist.
"""
import numpy as np
import pytest


@pytest.fixture
def small_X_y_data():
    def true_function(X):
        return np.sin(3 * X)

    def true_noise_scale(X):
        return np.abs(np.cos(X))

    n_samples = 100
    X = np.random.uniform(-2, 2, n_samples)
    y = true_function(X) + np.random.normal(scale=true_noise_scale(X), size=n_samples)
    return X[..., np.newaxis], y
