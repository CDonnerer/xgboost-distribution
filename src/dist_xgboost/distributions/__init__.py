from dist_xgboost.distributions.normal import Normal
from dist_xgboost.distributions.base import BaseDistribution


def get_distributions():
    """Get dict of all available distributions"""
    return {
        subclass.__name__.lower(): subclass
        for subclass in BaseDistribution.__subclasses__()
    }
