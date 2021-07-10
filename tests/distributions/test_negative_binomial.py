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
    grad, hess = negative_binomial.gradient_and_hessian(
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
