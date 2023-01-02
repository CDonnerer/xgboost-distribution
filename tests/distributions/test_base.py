"""Test the base distribution functionality
"""
import pytest

import numpy as np

from xgboost_distribution.distributions import (
    AVAILABLE_DISTRIBUTIONS,
    Normal,
    format_distribution_name,
    get_distribution,
    get_distribution_doc,
)


def test_get_distribution():
    name = "normal"
    normal_dist = get_distribution(name)
    assert isinstance(normal_dist, Normal)


def test_get_distribution_raises():
    name = "i-do-not-exist"
    with pytest.raises(ValueError):
        get_distribution(name)


def test_get_distribution_doc():
    distribution_doc = get_distribution_doc()
    assert isinstance(distribution_doc, str)


@pytest.mark.parametrize(
    "name, expected_name",
    [("LogNormal", "log-normal"), ("NegativeBinomial", "negative-binomial")],
)
def test_format_distribution_name(name, expected_name):
    formatted_name = format_distribution_name(name)
    assert formatted_name == expected_name


@pytest.mark.parametrize("distribution_name", AVAILABLE_DISTRIBUTIONS.keys())
def test_distribution_starting_params_shape(distribution_name):
    """We need to get as many starting params as distribution params"""
    y = np.random.choice(
        [
            1,
            2,
        ],
        5,
    )

    distribution = get_distribution(distribution_name)
    starting_params = distribution.starting_params(y=y)

    assert len(starting_params) == len(distribution.params)


# @pytest.mark.skip
@pytest.mark.parametrize("distribution_name", AVAILABLE_DISTRIBUTIONS.keys())
def test_distribution_loss_shape(distribution_name):
    """Ensure that evaluation fns return expect nd.array shape"""
    y = np.random.choice([1, 2], 5)  # fmt: off

    distribution = get_distribution(distribution_name)
    starting_params = distribution.starting_params(y=y)

    params = np.concatenate([starting_params]) * np.ones_like(y).reshape(-1, 1)

    if len(distribution.params) == 1:
        params = params.squeeze()  # for 1 param we get a squeezed array (n,) from xgb

    loss_name, loss = distribution.loss(y, params)

    assert isinstance(loss_name, str)
    assert loss.shape == y.shape
