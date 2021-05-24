"""Distribution base class
"""
from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    @abstractmethod
    def gradient_and_hessian(self, y, params):
        pass

    @abstractmethod
    def loss(self, y, params):
        pass

    @abstractmethod
    def predict(self, params):
        pass
