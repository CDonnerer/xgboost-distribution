"""Tests against data edge cases
"""
import pytest

import numpy as np

from xgboost_distribution.model import XGBDistribution


def generate_data_with_target_outliers(n_samples=30_000, n_outliers=100):
    """Generate data with small number of outliers in target

    Bulk of target is drawn from single distribution, with a *small* set of
    outliers. These inflate the variance for these data points, potentially
    causing overflow errors, if parameters are unconstrained.
    """
    y = np.concatenate(
        (
            np.random.normal(loc=3000, scale=500, size=n_samples - n_outliers),
            np.random.uniform(low=10_000, high=100_000, size=n_outliers),
        )
    )
    x = np.random.gamma(shape=2, scale=2, size=n_samples)
    return x[..., np.newaxis], y


def test_target_outlier_robust():
    X, y = generate_data_with_target_outliers()
    xgbd = XGBDistribution(
        distribution="normal",
        n_estimators=2,
    ).fit(X, y)

    # assert that we got here without issues
    assert isinstance(xgbd, XGBDistribution)


def generate_data_with_target_consts(n_samples=30_000, n_not_constant=100):
    """Generate data with constant/duplicates in target

    Having constant in target can cause variance to diminish, potentially
    causing overflow errors.
    """
    y = np.concatenate(
        (
            np.random.normal(loc=10, scale=5, size=n_not_constant),
            np.ones(shape=(n_samples - n_not_constant,)),
        )
    )
    x = np.concatenate((np.random.uniform(low=0, high=10, size=n_samples),))
    return x[..., np.newaxis], y


@pytest.mark.slow
def test_target_with_consts():
    X, y = generate_data_with_target_consts()
    xgbd = XGBDistribution(
        distribution="normal",
        n_estimators=100,
    ).fit(X, y)

    # assert that we got here without issues
    assert isinstance(xgbd, XGBDistribution)
