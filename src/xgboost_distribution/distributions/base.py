"""Distribution base class"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseDistribution(ABC):
    """Base class distribution for XGBDistribution.

    Note that distributions are stateless, hence a distribution is just a collection of
    functions that operate on the data (`y`) and the outputs of the xgboost (`params`).
    """

    def check_target(self, y) -> None:
        """Ensure that the target is compatible with the chosen distribution"""

    @property
    @abstractmethod
    def params(self) -> Tuple[str, ...]:
        """The parameter names of the distribution"""

    @abstractmethod
    def starting_params(self, y) -> Tuple[float, ...]:
        """The starting parameters of the distribution"""

    @abstractmethod
    def gradient_and_hessian(
        self, y, params, natural_gradient=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the gradient and hessian of the distribution"""

    @abstractmethod
    def loss(self, y, params) -> Tuple[str, np.ndarray]:
        """Evaluate the per sample loss (typically negative log-likelihood)"""

    @abstractmethod
    def predict(self, params) -> Tuple[np.ndarray, ...]:
        """Predict the parameters of a given distribution"""
