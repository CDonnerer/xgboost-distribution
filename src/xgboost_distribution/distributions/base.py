"""Distribution base class
"""
from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    """Base class distribution for XGBDistribution.

    Note that distributions are stateless, hence a distribution is just a collection of
    functions that operate on the data (`y`) and the outputs of the xgboost (`params`).
    """

    def check_target(self, y):
        """Ensure that the target is compatible with the chosen distribution"""

    @property
    @abstractmethod
    def params(self):
        """The parameter names of the distribution"""

    @abstractmethod
    def starting_params(self, y):
        """The starting parameters of the distribution"""

    @abstractmethod
    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Compute the gradient and hessian of the distribution"""

    @abstractmethod
    def loss(self, y, params):
        """Evaluate the per sample loss (typically negative log-likelihood)"""

    @abstractmethod
    def predict(self, params):
        """Predict the parameters of a given distribution"""
