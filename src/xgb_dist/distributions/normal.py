"""Normal distribution
"""
import numpy as np
from scipy.stats import norm

from xgb_dist.distributions.base import BaseDistribution
from collections import namedtuple


class Normal(BaseDistribution):
    """Implementation of normal distribution for XGBDistribution"""

    @property
    def params(self):
        return ("mean", "scale")

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        mean, log_scale = params[:, 0], params[:, 1]
        var = np.exp(2 * log_scale)

        grad = np.zeros(shape=(len(y), 2))
        grad[:, 0] = (mean - y) / var
        grad[:, 1] = 1 - ((y - mean) ** 2) / var

        if natural_gradient:
            fisher_matrix = np.zeros(shape=(len(y), 2, 2))
            fisher_matrix[:, 0, 0] = 1 / var
            fisher_matrix[:, 1, 1] = 2

            grad = np.linalg.solve(fisher_matrix, grad)

            hess = np.ones(shape=(len(y), 2))  # we set the hessian constant
        else:
            hess = np.zeros(shape=(len(y), 2))  # diagonal elements only
            hess[:, 0] = 1 / var
            hess[:, 1] = 2 * ((y - mean) ** 2) / var

        return grad, hess

    def loss(self, y, params):
        """Loss function to minimise"""
        mean, scale = self.predict(params)
        return "NormalError", -norm.logpdf(y, loc=mean, scale=scale).mean()

    def predict(self, params):
        mean, log_scale = params[:, 0], params[:, 1]

        log_scale = np.clip(log_scale, -100, 100)
        scale = np.exp(log_scale)

        Predictions = namedtuple("Predictions", (p for p in self.params))
        return Predictions(mean=mean, scale=scale)

    def starting_params(self, y):
        return np.mean(y), np.log(np.std(y))
