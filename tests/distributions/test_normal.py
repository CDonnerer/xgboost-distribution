import pytest

import numpy as np

from xgboost_distribution.distributions import Normal


@pytest.fixture
def normal():
    return Normal()


@pytest.mark.parametrize(
    "y, params, natural_gradient, expected_grad",
    [
        (
            np.array([0, 0]),
            np.array([[0, 1], [1, 0]]),
            True,
            np.array([[0, 0.5], [1, 0]]),
        ),
        (
            np.array([0, 0]),
            np.array([[0, 1], [1, 0]]),
            False,
            np.array([[0, 1], [1, 0]]),
        ),
    ],
)
def test_gradient_calculation(normal, y, params, natural_gradient, expected_grad):
    grad, _ = normal.gradient_and_hessian(y, params, natural_gradient=natural_gradient)
    np.testing.assert_array_equal(grad, expected_grad)
