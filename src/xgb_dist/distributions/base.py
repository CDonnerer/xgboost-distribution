"""Distribution base class
"""
from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    """Abstract base distribution for XGBDistribution.

    Note that by design all distributions are **stateless**.
    A distribution is thus a collection of functions that operate on the data
    (`y`) and the output of the xgboost Booster (`params`).
    """

    @property
    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def gradient_and_hessian(self, y, params):
        pass

    @abstractmethod
    def loss(self, y, params):
        pass

    @abstractmethod
    def predict(self, params):
        pass
