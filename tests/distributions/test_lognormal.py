import numpy as np
import pytest

from xgboost_distribution.distributions import LogNormal


@pytest.fixture
def lognormal():
    return LogNormal()


@pytest.mark.parametrize(
    "y, params, natural_gradient, expected_grad",
    [
        (
            np.array([1, 1]),
            np.array([[np.log(1), 2], [1, 0]]),
            True,
            np.array([[0, 0.5], [1, 0.0]]),
        ),
        (
            np.array([1, 1]),
            np.array([[np.log(1), 2], [1, 0]]),
            False,
            np.array([[0, 1], [1, 0.0]]),
        ),
    ],
)
def test_gradient_calculation(lognormal, y, params, natural_gradient, expected_grad):
    grad, hess = lognormal.gradient_and_hessian(
        y, params, natural_gradient=natural_gradient
    )
    np.testing.assert_array_equal(grad, expected_grad)
