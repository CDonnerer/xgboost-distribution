"""LogNormal distribution
"""
from collections import namedtuple

import numpy as np
from scipy.stats import lognorm

from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.utils import check_all_gt_zero

Predictions = namedtuple("Predictions", ("scale", "s"))


class LogNormal(BaseDistribution):
    """LogNormal distribution with log scoring.

    Definition:

        f(x) = exp( -[ (log(x) - log(scale)) / (2 s^2) ]^2 / 2 ) / s


    with parameters (scale, s).

    We reparameterize:
        s -> log(s) = a
        scale -> log(scale) = b

    Note that b essentially becomes the 'loc' of the distribution:

        log(x/scale) / s = ( log(x) - log(scale) ) / s

    which can then be taken analogous to the normal distribution's

        (x - loc) / scale

    Hence we can re-use the computations in `distribution.normal`, exchanging:

        y -> log(y)
        scale -> s

    """

    @property
    def params(self):
        return ("scale", "s")

    def check_target(self, y):
        check_all_gt_zero(y)

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        log_y = np.log(y)

        loc, log_s = self._split_params(params)  # note loc = log(scale)
        var = np.exp(2 * log_s)

        grad = np.zeros(shape=(len(y), 2))
        grad[:, 0] = (loc - log_y) / var
        grad[:, 1] = 1 - ((loc - log_y) ** 2) / var

        if natural_gradient:
            fisher_matrix = np.zeros(shape=(len(y), 2, 2))
            fisher_matrix[:, 0, 0] = 1 / var
            fisher_matrix[:, 1, 1] = 2

            grad = np.linalg.solve(fisher_matrix, grad)

            hess = np.ones(shape=(len(y), 2))  # we set the hessian constant
        else:
            hess = np.zeros(shape=(len(y), 2))  # diagonal elements only
            hess[:, 0] = 1 / var
            hess[:, 1] = 2 * ((log_y - loc) ** 2) / var

        return grad, hess

    def loss(self, y, params):
        scale, s = self.predict(params)
        return "LogNormal-NLL", -lognorm.logpdf(y, s=s, scale=scale)

    def predict(self, params):
        log_scale, log_s = self._split_params(params)
        scale, s = np.exp(log_scale), np.exp(log_s)

        return Predictions(scale=scale, s=s)

    def starting_params(self, y):
        log_y = np.log(y)
        return Predictions(scale=np.mean(log_y), s=np.log(np.std(log_y)))

    def _split_params(self, params):
        """Return log_scale (loc) and log_s from params"""
        return params[:, 0], params[:, 1]
