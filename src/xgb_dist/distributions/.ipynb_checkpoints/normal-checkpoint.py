"""Normal distribution
"""
import numpy as np
from scipy.stats import norm

from xgb_dist.distributions.base import BaseDistribution


class Normal(BaseDistribution):
    """Implementation of normal distribution for XGBDistribution"""

    @property
    def params(self):
        return ["mean", "var"]

    def gradient_and_hessian(self, y, params):
        """Gradient and diagonal hessian"""

        mean, log_var = params[:, 0], params[:, 1]
        var = (np.exp(log_var)) ** 2

        grad = np.zeros(shape=(len(y), 2))
        grad[:, 0] = (mean - y) / var
        grad[:, 1] = 1 - ((y - mean) ** 2) / var

        use_natural_grad = False
        if use_natural_grad:
            fisher_matrix = np.zeros(shape=(len(y), 2, 2))
            fisher_matrix[:, 0, 0] = 1 / var
            fisher_matrix[:, 1, 1] = 2

            grad = np.linalg.solve(fisher_matrix, grad)

            hess = np.zeros(shape=(len(y), 2))  # diagonal elements only
            hess[:, 0] = 1
            hess[:, 1] = 1

        else:
            hess = np.zeros(shape=(len(y), 2))  # diagonal elements only
            hess[:, 0] = 2 / var
            hess[:, 1] = 2

        return grad, hess

    def loss(self, y, params):
        """Loss function to minimise"""
        mean, var = self.predict(params)
        return "NormalError", -norm.logpdf(y, loc=mean, scale=var).mean()

    def predict(self, params):
        mean, log_var = params[:, 0], params[:, 1]

        log_var = np.clip(log_var, -100, 100)
        var = np.log(log_var)
        return mean, var ** 0.5
