"""Distribution base class
"""
from abc import ABC, abstractmethod
from collections import namedtuple


class BaseDistribution(ABC):
    """Base class distribution for XGBDistribution.

    Note that distributions are stateless, hence a distribution is just a collection of
    functions that operate on the data (`y`) and the outputs of the xgboost (`params`).
    """

    def __init__(self):
        self.Predictions = namedtuple("Predictions", (p for p in self.params))
        # hack to make pickling of namedtuple work
        globals()[self.Predictions.__name__] = self.Predictions

    def check_target(self, y):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def starting_params(self, y):
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
