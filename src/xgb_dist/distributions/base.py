"""Distribution base class

TODO:
- param names property

"""
from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    def __repr__(self):
        return f"`{self.__class__.__name__.lower()}`"

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
