"""
Task 4.2: Early Feature Fusion Pipeline (Per-Modality Z-Score & Calibrated SVM)

Procedural Steps (PDF Page 9):
1. Fit independent StandardScaler instances for each modality to eliminate scale dominance:
   z = (x - μ) / σ for face (576D), voice (46D), and print (64D) independently.
2. Concatenate normalized matrices horizontally into a single multimodal representation vector:
   X_fused = [Z_face || Z_voice || Z_print] ∈ R^(N x 686).
3. Train an Early Fusion SVM with RBF kernel and Platt probability calibration on the fused 686D vector.
"""

from typing import Tuple, Union

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from interfaces import (
    FACE_DIM,
    FUSED_DIM,
    PRINT_DIM,
    VOICE_DIM,
    BaseBiometricModel,
    BaseFeatureExtractor,
)


ModalityInput = Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _ensure_2d(arr: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != expected_dim:
        raise ValueError(f"{name} expected shape (N, {expected_dim}), got {x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return x


class EarlyFeatureFusionExtractor(BaseFeatureExtractor):
    """
    Fits 3 independent StandardScalers (Face 576D, Voice 46D, Fingerprint 64D)
    and concatenates the Z-score normalized matrices into a 686D float32 matrix.
    """

    def __init__(self) -> None:
        self.scaler_face = StandardScaler()
        self.scaler_voice = StandardScaler()
        self.scaler_print = StandardScaler()
        self.is_fitted_ = False

    @staticmethod
    def _unpack(samples: ModalityInput) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(samples, (tuple, list)):
            x_face, x_voice, x_print = samples
        elif isinstance(samples, np.ndarray):
            arr = _ensure_2d(samples, FUSED_DIM, "ConcatenatedModalities")
            x_face = arr[:, :FACE_DIM]
            x_voice = arr[:, FACE_DIM : FACE_DIM + VOICE_DIM]
            x_print = arr[:, FACE_DIM + VOICE_DIM :]
        else:
            raise TypeError("Expected (X_face, X_voice, X_print) tuple or (N, 686) array.")

        x_face = _ensure_2d(x_face, FACE_DIM, "X_face")
        x_voice = _ensure_2d(x_voice, VOICE_DIM, "X_voice")
        x_print = _ensure_2d(x_print, PRINT_DIM, "X_print")
        return x_face, x_voice, x_print

    def fit(self, samples: ModalityInput) -> "EarlyFeatureFusionExtractor":
        x_face, x_voice, x_print = self._unpack(samples)
        self.scaler_face.fit(x_face)
        self.scaler_voice.fit(x_voice)
        self.scaler_print.fit(x_print)
        self.is_fitted_ = True
        return self

    def transform(self, samples: ModalityInput) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("EarlyFeatureFusionExtractor must be fitted before transform().")
        x_face, x_voice, x_print = self._unpack(samples)
        z_face = self.scaler_face.transform(x_face).astype(np.float32)
        z_voice = self.scaler_voice.transform(x_voice).astype(np.float32)
        z_print = self.scaler_print.transform(x_print).astype(np.float32)
        return np.hstack([z_face, z_voice, z_print]).astype(np.float32)


class CalibratedSVM(BaseBiometricModel):
    """
    Classical RBF Support Vector Classifier with Platt scaling (probability=True).
    Used for unimodal baselines and for the 686D Early Fusion classifier.
    """

    def __init__(self, C: float = 10.0, gamma: str = "scale", use_scaler: bool = True, random_state: int = 42):
        self.C = C
        self.gamma = gamma
        self.use_scaler = use_scaler
        self.random_state = random_state
        self.scaler = StandardScaler() if use_scaler else None
        self.svc = SVC(
            C=C,
            kernel="rbf",
            gamma=gamma,
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )
        self.classes_ = None
        self.n_features_in_ = None

    def _prep(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self.n_features_in_ is not None and arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {arr.shape[1]}.")
        if self.scaler is not None:
            arr = self.scaler.fit_transform(arr) if fit else self.scaler.transform(arr)
        return arr.astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibratedSVM":
        arr = np.asarray(X, dtype=np.float32)
        self.n_features_in_ = arr.shape[1]
        X_scaled = self._prep(arr, fit=True)
        self.svc.fit(X_scaled, np.asarray(y, dtype=np.int64))
        self.classes_ = np.asarray(self.svc.classes_, dtype=np.int64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.svc.predict(self._prep(X)), dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.asarray(self.svc.predict_proba(self._prep(X)), dtype=np.float32)
        return probs / np.sum(probs, axis=1, keepdims=True)


class EarlyFusionMultimodalModel(BaseBiometricModel):
    """
    End-to-end Task 4.2 Multimodal Model:
    Combines EarlyFeatureFusionExtractor (3 independent StandardScalers -> 686D)
    with a Platt-calibrated RBF SVM.
    """

    def __init__(self, C: float = 10.0, gamma: str = "scale", random_state: int = 42):
        self.fusion_extractor = EarlyFeatureFusionExtractor()
        self.classifier = CalibratedSVM(C=C, gamma=gamma, use_scaler=False, random_state=random_state)
        self.classes_ = None

    def fit(self, X: ModalityInput, y: np.ndarray) -> "EarlyFusionMultimodalModel":
        X_fused = self.fusion_extractor.fit_transform(X)
        self.classifier.fit(X_fused, y)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, X: ModalityInput) -> np.ndarray:
        X_fused = self.fusion_extractor.transform(X)
        return self.classifier.predict(X_fused)

    def predict_proba(self, X: ModalityInput) -> np.ndarray:
        X_fused = self.fusion_extractor.transform(X)
        return self.classifier.predict_proba(X_fused)
