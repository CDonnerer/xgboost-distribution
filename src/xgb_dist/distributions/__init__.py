from xgb_dist.distributions.base import BaseDistribution
from xgb_dist.distributions.normal import Normal  # noqa

# TOTA: cleaner way of importing distribution subclasses?
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
            f"Distribution is not implemented! Please choose one of {AVAILABLE_DISTRIBUTIONS.keys()}"
        )

    return AVAILABLE_DISTRIBUTIONS[name]()


def get_distribution_doc():
    param_doc = """
    distribution : str, default='normal'
        Which distribution to estimate. Available choices:

    """
    for subclass in BaseDistribution.__subclasses__():
        param_doc += f"\t\t'{subclass.__name__.lower()}' : params = {subclass().params}"

    return param_doc
