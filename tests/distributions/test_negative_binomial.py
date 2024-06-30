import pytest

import numpy as np
import pandas as pd
from scipy.special import logit

from xgboost_distribution.distributions import NegativeBinomial


@pytest.fixture
def negative_binomial():
    return NegativeBinomial()


@pytest.mark.parametrize(
    "y, params, natural_gradient, expected_grad",
    [
        (
            np.array([1]),
            np.array([[np.log(2), logit(1.0)]]),
            True,
            np.array([[-1, 0.5]]),
        ),
    ],
)
def test_gradient_calculation(
    negative_binomial, y, params, natural_gradient, expected_grad
):
    grad, _ = negative_binomial.gradient_and_hessian(
        y, params, natural_gradient=natural_gradient
    )
    np.testing.assert_array_equal(grad, expected_grad)


def test_target_validation(negative_binomial):
    valid_target = np.array([0, 1, 4, 5, 10])
    negative_binomial.check_target(valid_target)


@pytest.mark.parametrize(
    "invalid_target",
    [np.array([-0.1, 1.2]), pd.Series([1.1, 0.4, 2.3])],
)
def test_target_validation_raises(negative_binomial, invalid_target):
    with pytest.raises(ValueError):
        negative_binomial.check_target(invalid_target)


@pytest.mark.parametrize(
    "y, params",
    [
        (
            np.array([20], dtype="float32"),
            np.array([[113.1, 11.2]], dtype="float32"),
        ),
        (
            np.array([20], dtype="float32"),
            np.array([[13.1, -111.2]], dtype="float32"),
        ),
    ],
)
def test_overflow_stability(negative_binomial, y, params):
    """Test stability against large/small values produced by xgboost"""
    grad, _ = negative_binomial.gradient_and_hessian(y, params)
    assert isinstance(grad, np.ndarray)

    n, p = negative_binomial.predict(params)
    assert all(np.isfinite(n))
    assert n.all()
    assert p.all()
