import pytest

import numpy as np
import pandas as pd

from xgboost_distribution.distributions import Exponential


@pytest.fixture
def exponential():
    return Exponential()


def test_target_validation(exponential):
    valid_target = np.array([0, 0.1, 1.4, 5.5, 12.3])
    exponential.check_target(valid_target)


@pytest.mark.parametrize(
    "invalid_target",
    [np.array([-0.1, 1.2]), pd.Series([-1.1, 0.4, 2.3])],
)
def test_target_validation_raises(exponential, invalid_target):
    with pytest.raises(ValueError):
        exponential.check_target(invalid_target)


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
            np.array([[0], [0.5]]),
        ),
    ],
)
def test_gradient_calculation(exponential, y, params, natural_gradient, expected_grad):
    grad, _ = exponential.gradient_and_hessian(
        y, params, natural_gradient=natural_gradient
    )
    np.testing.assert_array_equal(grad, expected_grad)


def test_loss(exponential):
    loss_name, loss_values = exponential.loss(
        # fmt: off
        y=np.array([1, ]),
        params=np.array([np.log(1), ]),
    )
    assert loss_name == "Exponential-NLL"
    np.testing.assert_array_equal(loss_values, np.array([1.0]))


@pytest.mark.parametrize(
    "y, params",
    [
        (
            np.array([34.7, 20.1], dtype="float32"),
            np.array([150, 20], dtype="float32"),
        ),
        (
            np.array([34.7, 20.1], dtype="float32"),
            np.array([-150, 20], dtype="float32"),
        ),
    ],
)
def test_overflow_stability(exponential, y, params):
    """Test stability against large/small values produced by xgboost"""

    # Scale should not have inf or zeros
    scale = exponential.predict(params)
    assert np.isfinite(scale).all()
    assert np.all(scale)
