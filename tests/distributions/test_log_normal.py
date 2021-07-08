import pytest

import numpy as np
import pandas as pd

from xgboost_distribution.distributions import LogNormal


@pytest.fixture
def lognormal():
    return LogNormal()


def test_target_validation(lognormal):
    valid_target = np.array([0, 1, 4, 5, 10])
    lognormal.check_target(valid_target)


@pytest.mark.parametrize(
    "invalid_target",
    [np.array([-0.1, 1.2]), pd.Series([-1.1, 0.4, 2.3])],
)
def test_target_validation_raises(lognormal, invalid_target):
    with pytest.raises(ValueError):
        lognormal.check_target(invalid_target)


@pytest.mark.parametrize(
    "y, params, natural_gradient, expected_grad",
    [
        (
            np.array([1, 1]),
            np.array([[np.log(1), 2], [1, 0]]),
            True,
            np.array([[0, 0.5], [1, 0]]),
        ),
        (
            np.array([1, 1]),
            np.array([[np.log(1), 2], [1, 0]]),
            False,
            np.array([[0, 1], [1, 0]]),
        ),
    ],
)
def test_gradient_calculation(lognormal, y, params, natural_gradient, expected_grad):
    grad, hess = lognormal.gradient_and_hessian(
        y, params, natural_gradient=natural_gradient
    )
    np.testing.assert_array_equal(grad, expected_grad)


def test_loss(lognormal):
    loss_name, loss_value = lognormal.loss(
        # fmt: off
        y=np.array([0, ]),
        params=np.array([[1, 0], ]),
    )
    assert loss_name == "LogNormalError"
    assert loss_value == np.inf
