"""Negative binomial distribution
"""
import numpy as np
from scipy.special import digamma, expit
from scipy.stats import nbinom

from xgboost_distribution.distributions.base import BaseDistribution


class NegativeBinomial(BaseDistribution):
    """Negative binomial distribution with log score

    Definition:

        f(k) = p^n (1 - p)^k binomial(n + k - 1, n - 1)

    with parameter (n, p), where n >= 0 and 1 >= p >= 0

    We reparameterize:
        n -> log(n) = a        |  e^a = n           |
        p -> log(p/(1-p)) = b  |  e^b = p / (1-p)   |  p = 1 / (1 + e^(-b))

    Thus the gradients become:

        d/da -log[f(k)] = -e^a [ digamma(k+e^a) - digamma(e^a) + log(p) ]
                        = -n * [ digamma(k+n) - digamma(n) + log(p) ]

        d/db -log[f(k)] = ( k * e^b - e^a ) / (e^b + 1)
                        = ( k - e^a * e^-b ) / (e^-b + 1)
                        = p *( k - e^a * e^-b )
                        = p * ( k - n e^-b)

    The Fisher Information = n / [ (1-p) p^2 ]

    In reparameterized form:

        n / [ (1-p) p^2 ] = I ( d/dn log(n) )^2 = I ( 1/n )^2
        n / [ (1-p) p^2 ] = I ( d/dp log(p/(1-p)) )^2 = I ( 1/(p-p^2) )^2

    We find:

        [  n**3 / [ (1-p) p^2 ], 0  ]
        [  0,  n / p ]

    Ref:
        https://www.wolframalpha.com/input/?i=d%2Fda+-log%28+%5B1+%2F+%281+%2B+e%5E%28-b%29%29%5D+%5E%28e%5Ea%29+%281+-+%5B1+%2F+%281+%2B+e%5E%28-b%29%29%5D%29%5Ek+binomial%28%28e%5Ea%29+%2B+k+-+1%2C+%28e%5Ea%29+-+1%29+%29


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
        grad[:, 1] = p * (y - n * np.exp(-raw_p))

        if natural_gradient:

            fisher_matrix = np.zeros(shape=(len(y), 2, 2))
            fisher_matrix[:, 0, 0] = n / (p * (1 - p) ** 2)
            fisher_matrix[:, 1, 1] = 1 / (n * p)

            grad = np.linalg.solve(fisher_matrix, grad)
            hess = np.ones(shape=(len(y), 2))  # we set the hessian constant

        else:
            raise NotImplementedError("TODO?")

        return grad, hess

    def loss(self, y, params):
        n, p = self.predict(params)
        return "NegativeBinomialError", -nbinom.logpmf(y, n=n, p=p).mean()

    def predict(self, params):
        log_n, raw_p = params[:, 0], params[:, 1]
        n = np.exp(log_n)
        p = expit(raw_p)
        return self.Predictions(n=n, p=p)

    def starting_params(self, y):
        return (np.log(np.mean(y)), -2)
