"""Negative binomial distribution
"""
import numpy as np
from scipy.special import digamma, expit
from scipy.stats import nbinom

from xgboost_distribution.distributions.base import BaseDistribution


class NegativeBinomial(BaseDistribution):
    """Negative binomial distribution

    We reparameterize from eta -> theta, such that:

        eta = (n, p)
        theta = (log(n), log(p/(1-p)))

    Jacobian = [
        [1/n, 0],
        [0, 1/(p - p^2)]
    ]
    J^(-1) = [
        [n, 0],
        [0, p - p^2]
    ]

    I_eta =
        [n / [(1 - p)^2 * p], 0],
        [0, n / [(1 - p)^2 * p]]

        =
    [n,       0] [    , 0] [n,       0  ]
    [0, p - p^2] [0, n / [(1 - p)^2 * p]] [0, p - p^2]


    (x_1) = n / [(1 - p)^2 * p] / n^2
          = 1 / (n * p * (1-p)^2 )

    (x_2) = n / [(1 - p)^2 * p] / (p - p^2)^2


    I_eta = J^(-1) I_theta J^(-1)



    """

    @property
    def params(self):
        return ("n", "p")

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        log_n, raw_p = params[:, 0], params[:, 1]
        n = np.exp(log_n)
        p = expit(raw_p)

        grad = np.zeros(shape=(len(y), 2))

        grad[:, 0] = -n * (digamma(y + n) - digamma(n) + np.log(p))
        grad[:, 1] = p * (y * np.exp(raw_p) - n)

        if natural_gradient:

            fisher_matrix = np.zeros(shape=(len(y), 2, 2))
            fisher_matrix[:, 0, 0] = 1 / (p * (1 - p) ** 2)
            fisher_matrix[:, 1, 1] = (n * p ** 2) / (1 - p)
            # fisher_matrix[:, 0, 0] = 1 / (n * p * (1 - p) ** 2)
            # fisher_matrix[:, 1, 1] = n / (p ** 4 -

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
