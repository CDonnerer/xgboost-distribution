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
    dists = {
        "exponential": scipy.stats.expon.logpdf,
        "laplace": scipy.stats.laplace.logpdf,
        "log-normal": scipy.stats.lognorm.logpdf,
        "negative-binomial": scipy.stats.nbinom.logpmf,
        "normal": scipy.stats.norm.logpdf,
        "poisson": scipy.stats.poisson.logpmf,
    }

    def score_func(y, y_pred):
        return dists[distribution](y, *y_pred).mean()

    return score_func
