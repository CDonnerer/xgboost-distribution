"""Poisson distribution
"""
import numpy as np
from scipy.stats import poisson

from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.utils import check_is_integer, check_is_positive


class Poisson(BaseDistribution):
    """Poisson distribution with log score

    Definition:

        f(k) = e^(-mu) mu^k / k!

    We reparameterize mu -> log(mu) = a to ensure mu >= 0. Gradient:

        d/da -log[f(k)] = e^a - k  = mu - k

    The Fisher information = 1 / mu, which needs to be expressed in the
    reparameterized form:

        1 / mu = I ( d/dmu log(mu) )^2 = I ( 1/ mu )^2

    Hence we find: I = mu

    """

    @property
    def params(self):
        return ("mu",)

    def check_target(self, y):
        check_is_integer(y)
        check_is_positive(y)

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        (mu,) = self.predict(params)

        grad = np.zeros(shape=(len(y), 1))
        grad[:, 0] = mu - y

        if natural_gradient:
            fisher_matrix = np.zeros(shape=(len(y), 1, 1))
            fisher_matrix[:, 0, 0] = mu

            grad = np.linalg.solve(fisher_matrix, grad)

            hess = np.ones(shape=(len(y), 1))  # we set the hessian constant
        else:
            hess = mu

        return grad, hess

    def loss(self, y, params):
        mu = self.predict(params)
        return "PoissonError", -poisson.logpmf(y, mu=mu).mean()

    def predict(self, params):
        log_mu = params
        mu = np.exp(log_mu)
        return self.Predictions(mu=mu)

    def starting_params(self, y):
        return (np.log(np.mean(y)),)
