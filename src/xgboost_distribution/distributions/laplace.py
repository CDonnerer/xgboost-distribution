"""Laplace distribution
"""
from collections import namedtuple

import numpy as np
from scipy.stats import cauchy, laplace

from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.utils import safe_exp

Params = namedtuple("Params", ("loc", "scale"))


class Laplace(BaseDistribution):
    """Laplace distribution with log scoring

    Definition:

        f(x) = 1/2 * exp( -| (x - loc) / scale | ) / scale

    We reparameterize:
        loc   -> loc = a
        scale -> log(scale) = b to ensure scale >= 0.

    Thus:

        f(x) = 1/2 * exp( -| (x - a) / e^b | ) / e^b

    We compute the gradients:

        d/da -log[f(x)] = (a - x) / (scale * | a-x | )
        d/db -log[f(x)] = 1 - | a-x | / scale

    To second order:

        d2/da2 -log[f(x)] = 2 δ(x-a) / scale
        d2/db2 -log[f(x)] = | a-x | / scale

    The Fisher information is:

        I(loc) = 1 / scale^2
        I(scale) = 1 / scale^2

    which needs to be expressed in reparameterized form:

        1 / scale^2 = I_r(loc) (d/d(loc) loc) ^2
                    = I_r(loc)

        1 / scale^2 = I_r(scale) ( d/d(scale) log(scale) )^2
                    = I_r(scale) ( 1/ scale )^2

    Hence:

        I_r(loc) = 1 / scale^2
        I_r(scale) = 1

    """

    @property
    def params(self):
        return Params._fields

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        loc, scale = self.predict(params)

        grad = np.zeros(shape=(len(y), 2), dtype="float32")
        grad[:, 0] = np.sign(loc - y) / scale
        grad[:, 1] = 1 - np.abs(loc - y) / scale

        if natural_gradient:
            fisher_matrix = np.zeros(shape=(len(y), 2, 2), dtype="float32")
            fisher_matrix[:, 0, 0] = 1 / scale**2
            fisher_matrix[:, 1, 1] = 1

            grad = np.linalg.solve(fisher_matrix, grad)
            hess = np.ones(shape=(len(y), 2), dtype="float32")  # constant hessian
        else:
            hess = np.zeros(shape=(len(y), 2), dtype="float32")
            # Note: Delta functions won't work well, hence we approximate with cauchy
            hess[:, 0] = 2 * cauchy.pdf(y, loc, scale) / scale
            hess[:, 1] = 1 - grad[:, 1]

        return grad, hess

    def loss(self, y, params):
        loc, scale = self.predict(params)
        return "Laplace-NLL", -laplace.logpdf(y, loc=loc, scale=scale)

    def predict(self, params):
        loc, log_scale = params[:, 0], params[:, 1]
        scale = safe_exp(log_scale)
        return Params(loc=loc, scale=scale)

    def starting_params(self, y):
        return Params(loc=np.mean(y), scale=np.log(np.std(y)))
