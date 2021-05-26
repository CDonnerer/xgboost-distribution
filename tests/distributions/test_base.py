"""Test the base distribution
"""

from xgb_dist.distributions import get_distribution_doc


def test_get_distribution():
    distribution_doc = get_distribution_doc()
    assert distribution_doc
