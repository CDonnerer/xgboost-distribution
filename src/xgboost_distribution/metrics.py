"""Metrics for evaluating xgboost-distribution scores
"""
from typing import Callable, Tuple

import numpy as np
import scipy
from xgboost._typing import ArrayLike


def get_ll_score_func(
    distribution: str,
) -> Callable[[ArrayLike, Tuple[np.ndarray]], float]:
    """Get log-likelihood scoring function for a given distribution

    Parameters
    ----------
    distribution : str

    Returns
    -------
    Callable
        Scoring function
    """
    dist_ll = {
        "exponential": scipy.stats.expon.logpdf,
        "laplace": scipy.stats.laplace.logpdf,
        "log-normal": scipy.stats.lognorm.logpdf,
        "negative-binomial": scipy.stats.nbinom.logpmf,
        "normal": scipy.stats.norm.logpdf,
        "poisson": scipy.stats.poisson.logpmf,
    }

    def score_func(y, y_pred):
        return dist_ll[distribution](y, *y_pred).mean()

    return score_func


def wrap_point_estimate_score_func(
    score_func: Callable, distribution: str
) -> Callable[[ArrayLike, Tuple[np.ndarray]], float]:
    """Wrap a point estimate scoring function for scoring with xgboost-distribution

    Parameters
    ----------
    score_func : Callable
    distribution : str

    Returns
    -------
    Callable[[ArrayLike, Tuple[np.ndarray]], float]
        Scoring function
    """
    dist_point_estimator = {
        "normal": lambda x: x.loc,
        # TODO: What about the others?
    }

    def new_score_func(y, y_pred, **kwargs):
        y_pred = dist_point_estimator[distribution](y_pred)
        return score_func(y, y_pred, **kwargs)

    return new_score_func
