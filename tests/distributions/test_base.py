"""Test the base distribution functionality
"""
import pytest

from xgb_dist.distributions import Normal, get_distribution, get_distribution_doc


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
