"""Test suite for metrics
"""

from xgboost_distribution.distributions import AVAILABLE_DISTRIBUTIONS
from xgboost_distribution.metrics import get_ll_score_func


def test_get_ll_score_func_distribution_exist():
    for distribution in AVAILABLE_DISTRIBUTIONS.keys():
        score_func = get_ll_score_func(distribution=distribution)
        assert callable(score_func)
