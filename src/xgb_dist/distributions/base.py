"""Distribution base class
"""
from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    def __repr__(self):
        return self.__class__.__name__.lower()

    @abstractmethod
    def gradient_and_hessian(self, y, params):
        pass

    @abstractmethod
    def loss(self, y, params):
        pass

    @abstractmethod
    def predict(self, params):
        pass
