from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        tbd
        """
        pass

    @abstractmethod
    def predict_with_uncertainties(
        self, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        tbd
        """
        pass
