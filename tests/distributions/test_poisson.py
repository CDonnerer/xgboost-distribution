import numpy as np
import pandas as pd
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


def test_target_validation(poisson):
    valid_target = np.array([0, 1, 4, 5, 10])
    poisson.check_target(valid_target)


@pytest.mark.parametrize(
    "invalid_target",
    [np.array([-0.1, 1.2]), pd.Series([1.1, 0.4, 2.3])],
)
def test_target_validation_raises(poisson, invalid_target):
    with pytest.raises(ValueError):
        poisson.check_target(invalid_target)
