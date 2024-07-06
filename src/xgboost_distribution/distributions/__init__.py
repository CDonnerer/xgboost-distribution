import re

from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.exponential import Exponential  # noqa
from xgboost_distribution.distributions.laplace import Laplace  # noqa
from xgboost_distribution.distributions.log_normal import LogNormal  # noqa
from xgboost_distribution.distributions.negative_binomial import (  # noqa
    NegativeBinomial,
)
from xgboost_distribution.distributions.normal import Normal  # noqa
from xgboost_distribution.distributions.poisson import Poisson  # noqa

# TOTA: alternative way of importing distribution subclasses?
# __all__ = ["normal"]
# from . import *


def format_distribution_name(class_name):
    return re.sub(r"(?<!^)(?=[A-Z])", "-", class_name).lower()


AVAILABLE_DISTRIBUTIONS = {
    format_distribution_name(subclass.__name__): subclass
    for subclass in BaseDistribution.__subclasses__()
}


def get_distribution(name):
    """Get instantianted distribution based on name"""

    if name not in AVAILABLE_DISTRIBUTIONS:
        raise ValueError(
            "Distribution is not implemented! Please choose one of "
            f"{set(AVAILABLE_DISTRIBUTIONS.keys())}"
        )

    return AVAILABLE_DISTRIBUTIONS[name]()


def get_distribution_doc():
    """Construct docstring for `distribution` param in XGBDistribution model"""

    param_doc = f"""
    distribution : {set(AVAILABLE_DISTRIBUTIONS.keys())}, default='normal'
        Which distribution to estimate. Available choices:
    """

    for name, subclass in AVAILABLE_DISTRIBUTIONS.items():
        param_doc += f"""
                - \"{name}\" - parameters: {subclass().params}\n"""

    param_doc += """
        Please see `scipy.stats` for a full description of the parameters of
        each distribution: https://docs.scipy.org/doc/scipy/reference/stats.html

        Note that distributions are fit using Maximum Likelihood Estimation, which
        internally corresponds to minimising the negative log likelihood (NLL).
    """

    return param_doc
