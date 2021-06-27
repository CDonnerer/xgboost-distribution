"""Normal distribution
"""
import numpy as np
from scipy.stats import norm

from xgboost_distribution.distributions.base import BaseDistribution


class Normal(BaseDistribution):
    """Normal distribution with log scoring rule

    We estimate two parameters, say a and b, such that:
        - a = mean
        - b = log ( variance ** (1/2) )

    where mean, variance are the parameters of the normal distribution.

    Note that we follow the `scipy.stats.norm` notation where:
        - loc = mean
        - scale = standard deviation
    """

    @property
    def params(self):
        return ("loc", "scale")

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        loc, log_scale = self._split_params(params)
        var = np.exp(2 * log_scale)

        grad = np.zeros(shape=(len(y), 2))
        grad[:, 0] = (loc - y) / var
        grad[:, 1] = 1 - ((y - loc) ** 2) / var

        if natural_gradient:
            fisher_matrix = np.zeros(shape=(len(y), 2, 2))
            fisher_matrix[:, 0, 0] = 1 / var
            fisher_matrix[:, 1, 1] = 2

            grad = np.linalg.solve(fisher_matrix, grad)

            hess = np.ones(shape=(len(y), 2))  # we set the hessian constant
        else:
            hess = np.zeros(shape=(len(y), 2))  # diagonal elements only
            hess[:, 0] = 1 / var
            hess[:, 1] = 2 * ((y - loc) ** 2) / var

        return grad, hess

    def loss(self, y, params):
        loc, scale = self.predict(params)
        return "NormalError", -norm.logpdf(y, loc=loc, scale=scale).mean()

    def predict(self, params):
        loc, log_scale = self._split_params(params)
        # log_scale = np.clip(log_scale, -100, 100)  # TODO: is this needed?
        scale = np.exp(log_scale)

        return self.Predictions(loc=loc, scale=scale)

    def starting_params(self, y):
        return np.mean(y), np.log(np.std(y))

    def _split_params(self, params):
        """Return loc and log_scale from params"""
        return params[:, 0], params[:, 1]
