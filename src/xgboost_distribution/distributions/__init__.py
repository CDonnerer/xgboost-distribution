from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.normal import Normal  # noqa

# TOTA: alternative way of importing distribution subclasses?
# __all__ = ["normal"]
# from . import *


AVAILABLE_DISTRIBUTIONS = {
    subclass.__name__.lower(): subclass
    for subclass in BaseDistribution.__subclasses__()
}


def get_distribution(name):
    """Get instantianted distribution based on name"""

    if name not in AVAILABLE_DISTRIBUTIONS.keys():
        raise ValueError(
            "Distribution is not implemented! Please choose one of "
            f"{set(AVAILABLE_DISTRIBUTIONS.keys())}"
        )

    return AVAILABLE_DISTRIBUTIONS[name]()


def get_distribution_doc():
    """Construct docstring for `distribution` param in XGBDistribution model"""

    param_doc = f"""
    distribution : {str(set(AVAILABLE_DISTRIBUTIONS.keys()))}, default='normal'
        Which distribution to estimate. Available choices:
    """

    for name, subclass in AVAILABLE_DISTRIBUTIONS.items():
        param_doc += f"""
                '{name}' : parameters = {subclass().params}"""

    return param_doc
