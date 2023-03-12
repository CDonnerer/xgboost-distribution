"""Negative binomial distribution
"""
from collections import namedtuple

import numpy as np
from scipy.special import digamma, expit
from scipy.stats import nbinom

from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.utils import (
    MAX_EXPONENT,
    MIN_EXPONENT,
    check_all_ge_zero,
    check_all_integer,
    safe_exp,
)

Params = namedtuple("Params", ("n", "p"))


class NegativeBinomial(BaseDistribution):
    """Negative binomial distribution with log score

    Definition:

        f(k) = p^n (1 - p)^k binomial(n + k - 1, n - 1)

    with parameter (n, p), where n >= 0 and 1 >= p >= 0

    We reparameterize:

        n -> log(n) = a        |  e^a = n
        p -> log(p/(1-p)) = b  |  e^b = p / (1-p)   |  p = 1 / (1 + e^-b)

    The gradients are:

        d/da -log[f(k)] = -e^a [ digamma(k+e^a) - digamma(e^a) + log(p) ]
                        = -n   [ digamma(k+n) - digamma(n) + log(p) ]

        d/db -log[f(k)] = (k e^b - e^a) / (e^b + 1)
                        = (k - e^a e^-b) / (e^-b + 1)
                        = p * (k - e^a e^-b)
                        = p * (k - n e^-b)

    The Fisher Information:

        I(n) ~ p / [ n (p+1) ]
        I(p) = n / [ p (1-p)^2 ]

    where we used an approximation for I(n) presented here:
        http://erepository.uonbi.ac.ke:8080/xmlui/handle/123456789/33803

    In reparameterized form, we find I_r(n) and I_r(p):

        p / [ n (p+1) ] = I_r(n) [ d/dn log(n) ]^2
                        = I_r(n) ( 1/n )^2

        -> I_r(n) = np / (p+1)

        n / [ p (1-p)^2 ] = I_r(p) [ d/dp log(p/(1-p)) ]^2
                          = I_r(p) ( 1/ [ p (1-p) ] )^2

        -> I_r(p) = [ p^2 (1-p)^2 n ] / [ p (1-p)^2 ] = np

    Hence the reparameterized Fisher information:

        [  np / (p+1), 0 ]
        [  0,         np ]

    Ref:

        https://www.wolframalpha.com/input/?i=d%2Fda+-log%28+%5B1+%2F+%281+%2B+e%5E%28-b%29%29%5D+%5E%28e%5Ea%29+%281+-+%5B1+%2F+%281+%2B+e%5E%28-b%29%29%5D%29%5Ek+binomial%28%28e%5Ea%29+%2B+k+-+1%2C+%28e%5Ea%29+-+1%29+%29

    """

    @property
    def params(self):
        return Params._fields

    def check_target(self, y):
        check_all_integer(y)
        check_all_ge_zero(y)

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        n, p = self.predict(params)

        grad = np.zeros(shape=(len(y), 2), dtype="float32")

        grad[:, 0] = -n * (digamma(y + n) - digamma(n) + np.log(p))
        grad[:, 1] = p * (y - n * (1 - p) / p)

        if natural_gradient:
            fisher_matrix = np.zeros(shape=(len(y), 2, 2), dtype="float32")
            fisher_matrix[:, 0, 0] = (n * p) / (p + 1)
            fisher_matrix[:, 1, 1] = n * p

            grad = np.linalg.solve(fisher_matrix, grad)
            hess = np.ones(shape=(len(y), 2), dtype="float32")  # constant hessian
        else:
            raise NotImplementedError(
                "Normal gradients are currently not supported by this "
                "distribution. Please use natural gradients!"
            )

        return grad, hess

    def loss(self, y, params):
        n, p = self.predict(params)
        return "NegativeBinomial-NLL", -nbinom.logpmf(y, n=n, p=p)

    def predict(self, params):
        log_n, logits = params[:, 0], params[:, 1]

        n = safe_exp(log_n)
        logits = np.clip(logits, a_min=MIN_EXPONENT, a_max=MAX_EXPONENT)

        p = expit(logits)
        return Params(n=n, p=p)

    def starting_params(self, y):
        # TODO: starting params can matter a lot?
        return Params(n=np.log(np.mean(y)), p=0)  # expit(0) = 0.5
