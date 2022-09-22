"""Metrics for evaluating xgboost-distribution scores
"""
import scipy


def get_ll_score_func(distribution: str):
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
