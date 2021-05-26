from xgb_dist.distributions.base import BaseDistribution
from xgb_dist.distributions.normal import Normal  # noqa


def get_distributions():
    """Get dict of all available distributions"""
    return {
        subclass.__name__.lower(): subclass
        for subclass in BaseDistribution.__subclasses__()
    }


def get_distribution_doc():
    param_doc = """
    distribution : str, default='normal'
        Which distribution to estimate. Available choices:

    """
    for subclass in BaseDistribution.__subclasses__():
        param_doc += f"\t\t'{subclass.__name__.lower()}' : params = {subclass().params}"

    return param_doc
