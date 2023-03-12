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


@pytest.mark.parametrize(
    "y, params",
    [
        (
            np.array([34.7, 20.1], dtype="float32"),
            np.array([[31.2, 50], [11, 6.7]], dtype="float32"),
        ),
        (
            np.array([34.7, 20.1], dtype="float32"),
            np.array([[1.2e3, -100], [11, 6.7]], dtype="float32"),
        ),
    ],
)
def test_overflow_stability(normal, y, params):
    """Test stability against large/small values produced by xgboost"""

    grad, _ = normal.gradient_and_hessian(y, params)
    assert isinstance(grad, np.ndarray)

    # Var should not have inf or zeros
    _, scale = normal.predict(params)
    var = scale**2
    assert all(np.isfinite(var))
    assert np.all(var)
