"""Poisson distribution
"""
import numpy as np
from scipy.stats import poisson

from xgboost_distribution.distributions.base import BaseDistribution


class Poisson(BaseDistribution):
    """Poisson distribution

    The Fisher information:

        I_eta = 1 / mu

    defining eta = (mu,) and theta = (log(mu),), then

        I_eta = I_theta (d theta / d eta)^ 2

        (d theta / d eta)^2  = (1 / mu)^2

    thus

        I_theta = I_eta mu ^2 = mu

    """

    @property
    def params(self):
        return ("mu",)

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        log_mu = params
        mu = np.exp(log_mu)

        grad = np.zeros(shape=(len(y), 1))
        grad[:, 0] = mu - y

        if natural_gradient:
            fisher_matrix = np.zeros(shape=(len(y), 1, 1))
            fisher_matrix[:, 0, 0] = mu

            grad = np.linalg.solve(fisher_matrix, grad)

            hess = np.ones(shape=(len(y), 1))  # we set the hessian constant
        else:
            raise NotImplementedError("TODO")

        return grad, hess

    def loss(self, y, params):
        mu = self.predict(params)
        return "PoissonError", -poisson.logpmf(y, mu=mu).mean()

    def predict(self, params):
        log_mu = params
        mu = np.exp(log_mu)

        return self.Predictions(mu=mu)

    def starting_params(self, y):
        return (np.mean(y),)
