"""
src/base_interfaces.py
Zero-blocking base interface contracts for all biometric modalities.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extraction modules."""

    @abstractmethod
    def fit(self, samples: list) -> "BaseFeatureExtractor":
        """Fit extractor parameters on training samples."""
        pass

    @abstractmethod
    def transform(self, samples: list) -> np.ndarray:
        """
        Transform raw input samples into 1D feature vectors.
        Returns:
            np.ndarray of shape (N, D) and dtype float32.
        """
        pass

    def fit_transform(self, samples: list) -> np.ndarray:
        return self.fit(samples).transform(samples)


class BaseBiometricModel(ABC):
    """Abstract base class for all biometric classifier models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseBiometricModel":
        """Fit the model on extracted feature matrix X and labels y."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict integer class labels for feature matrix X."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict calibrated class probabilities for feature matrix X.
        Returns:
            np.ndarray of shape (N, num_classes) where each row sums to 1.0.
        """
        pass