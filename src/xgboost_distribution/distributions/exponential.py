"""Exponential distribution
"""
import numpy as np
from scipy.stats import expon

from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.utils import check_all_ge_zero


class Exponential(BaseDistribution):
    """Exponential distribution with log score

    Definition:

        f(x) = 1 / scale * e^(-x / scale)

    We reparameterize scale -> log(scale) = a to ensure scale >= 0. Gradient:

        d/da -log[f(x)] = d/da -log[1/e^a e^(-x / e^a)]
                        = 1 - x e^-a
                        = 1 - x / scale

    The Fisher information = 1 / scale^2, when reparameterized:

        1 / scale^2 = I ( d/d(scale) log(scale) )^2 = I ( 1/ scale )^2

    Hence we find: I = 1

    """

    @property
    def params(self):
        return ("scale",)

    def check_target(self, y):
        check_all_ge_zero(y)

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        (scale,) = self.predict(params)

        grad = np.zeros(shape=(len(y), 1))
        grad[:, 0] = 1 - y / scale

        if natural_gradient:
            fisher_matrix = np.ones(shape=(len(y), 1, 1))

            grad = np.linalg.solve(fisher_matrix, grad)
            hess = np.ones(shape=(len(y), 1))  # we set the hessian constant
        else:
            hess = -(grad - 1)

        return grad, hess

    def loss(self, y, params):
        scale = self.predict(params)
        return "Exponential-NLL", -expon.logpdf(y, scale=scale).mean()

    def predict(self, params):
        log_scale = params
        scale = np.exp(log_scale)
        return self.Predictions(scale=scale)

    def starting_params(self, y):
        return (np.log(np.mean(y)),)
