"""Negative binomial distribution
"""
import numpy as np
from scipy.special import digamma, expit
from scipy.stats import nbinom

from xgboost_distribution.distributions.base import BaseDistribution


class NegativeBinomial(BaseDistribution):
    """Negative binomial distribution"""

    @property
    def params(self):
        return ("n", "p")

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        n, p = self.predict(params)

        grad = np.zeros(shape=(len(y), 2))

        grad[:, 0] = -n * (digamma(y + n) - digamma(n) + np.log(p))
        grad[:, 1] = -n / p - y / (p - 1)

        if natural_gradient:

            # TODO: the below can't be right as it's straight from wikipedia
            fisher_matrix = np.zeros(shape=(len(y), 2, 2))
            fisher_matrix[:, 0, 0] = n / (p * (1 - p) ** 2)
            fisher_matrix[:, 1, 1] = n / (p * (1 - p) ** 2)

            grad = np.linalg.solve(fisher_matrix, grad)

        hess = np.ones(shape=(len(y), 2))  # we set the hessian constant

        return grad, hess

    def loss(self, y, params):
        n, p = self.predict(params)
        return "NegativeBinomialError", -nbinom.logpmf(y, n=n, p=p).mean()

    def predict(self, params):
        log_n, raw_p = params[:, 0], params[:, 1]
        n = np.exp(log_n)
        # eps = 1e-9
        # p = np.clip(expit(raw_p), a_min=eps, a_max=1 - eps)
        p = expit(raw_p)

        return self.Predictions(n=n, p=p)

    def starting_params(self, y):
        return (np.log(np.mean(y)), 0.0)  # expit(0)=0.5
