"""
Task 4.1: Zero-Blocking Interface Contracts (interfaces.py)

Defines the abstract base classes required across all 4 team branches:
1. BaseFeatureExtractor : enforces fit(samples) and transform(samples) -> np.ndarray
2. BaseBiometricModel   : enforces fit(X, y), predict(X), and predict_proba(X) -> np.ndarray
3. DecisionRecord       : structured output for the 3-stage hierarchical pipeline
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np

# Modality dimensions from Project Plan
FACE_DIM = 576       # HOG (Task 1.2)
VOICE_DIM = 46       # MFCC + ZCR + RMS (Task 2.2)
PRINT_DIM = 64       # 2D Gabor Filter Bank (Task 3.2)
FUSED_DIM = 686      # 576 + 46 + 64 (Task 4.2)

IMPOSTOR_CLASS = 0

STATUS_VERIFIED = "VERIFIED"
STATUS_REJECTED_IMPOSTOR = "REJECTED_IMPOSTOR"
STATUS_PERMISSION_DENIED = "PERMISSION_DENIED"


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors."""

    @abstractmethod
    def fit(self, samples: Any) -> "BaseFeatureExtractor":
        """Fit extractor parameters on training samples."""
        pass

    @abstractmethod
    def transform(self, samples: Any) -> np.ndarray:
        """Transform samples into a 2D float32 numpy array of shape (N, D)."""
        pass

    def fit_transform(self, samples: Any) -> np.ndarray:
        """Fit on samples and return the transformed feature matrix."""
        return self.fit(samples).transform(samples)

    def save(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseFeatureExtractor":
        return joblib.load(Path(path))


class BaseBiometricModel(ABC):
    """Abstract base class for all biometric classifiers."""

    @abstractmethod
    def fit(self, X: Any, y: np.ndarray) -> "BaseBiometricModel":
        """Train classifier on feature matrix X and labels y."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for feature matrix X."""
        pass

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict calibrated posterior probabilities P(Y=k|X) summing to 1.0 per row."""
        pass

    def save(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseBiometricModel":
        return joblib.load(Path(path))


@dataclass
class DecisionRecord:
    """Structured decision output for Task 4.3 (Status, Identity, Command, Stage, Confidence)."""

    status: str
    identity: int
    command: str
    stage: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key.lower())
