"""LogNormal distribution
"""
import numpy as np
from scipy.stats import lognorm

from xgboost_distribution.distributions.base import BaseDistribution


class LogNormal(BaseDistribution):
    """LogNormal distribution"""

    @property
    def params(self):
        return ("s", "scale")

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        log_y = np.log(y)

        loc, log_scale = self._split_params(params)
        var = np.exp(2 * log_scale)

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
        s, scale = self.predict(params)
        return "LogNormalError", -lognorm.logpdf(y, s=s, scale=scale).mean()

    def predict(self, params):
        loc, log_scale = self._split_params(params)
        s = np.exp(log_scale)
        scale = np.exp(loc)

        return self.Predictions(s=s, scale=scale)

    def starting_params(self, y):
        log_y = np.log(y)

        return np.mean(log_y), np.log(np.std(log_y))

    def _split_params(self, params):
        """Return loc and log_scale from params"""
        return params[:, 0], params[:, 1]
