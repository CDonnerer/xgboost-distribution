import numpy as np
import pytest

from xgboost_distribution.distributions import Poisson


@pytest.fixture
def poisson():
    return Poisson()


@pytest.mark.parametrize(
    "y, params, natural_gradient, expected_grad",
    [
        (
            np.array([1, 1]),
            np.array([np.log(1), np.log(2)]),
            True,
            np.array([[0], [0.5]]),
        ),
        (
            np.array([1, 1]),
            np.array([np.log(1), np.log(2)]),
            False,
            np.array([[0], [1.0]]),
        ),
    ],
)
def test_gradient_calculation(poisson, y, params, natural_gradient, expected_grad):
    grad, hess = poisson.gradient_and_hessian(
        y, params, natural_gradient=natural_gradient
    )
    np.testing.assert_array_equal(grad, expected_grad)
