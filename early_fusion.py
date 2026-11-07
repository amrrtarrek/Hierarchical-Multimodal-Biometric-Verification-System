"""
Task 4.2: Early Feature Fusion Pipeline (early_fusion.py)

Procedural Steps (PDF Page 9):
1. Fit independent StandardScaler instances for each modality to eliminate scale dominance:
   z = (x - μ) / σ for face (576D), voice (46D), and print (64D) independently.
2. Concatenate normalized matrices horizontally into a single 686D multimodal vector:
   X_fused = [Z_face || Z_voice || Z_print] ∈ R^(N x 686).
3. Train an Early Fusion SVM with RBF kernel and Platt probability calibration on the 686D vector.
"""

from typing import Tuple

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

ModalityTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]


class EarlyFeatureFusionExtractor(BaseFeatureExtractor):
    """
    Fits 3 independent StandardScalers (Face 576D, Voice 46D, Fingerprint 64D)
    and concatenates the Z-score normalized matrices horizontally into (N, 686).
    """

    def __init__(self) -> None:
        self.scaler_face = StandardScaler()
        self.scaler_voice = StandardScaler()
        self.scaler_print = StandardScaler()
        self.is_fitted_ = False

    @staticmethod
    def _prepare_modalities(samples: ModalityTuple) -> ModalityTuple:
        x_face, x_voice, x_print = samples
        x_face = np.atleast_2d(np.asarray(x_face, dtype=np.float32))
        x_voice = np.atleast_2d(np.asarray(x_voice, dtype=np.float32))
        x_print = np.atleast_2d(np.asarray(x_print, dtype=np.float32))

        if x_face.shape[1] != FACE_DIM or x_voice.shape[1] != VOICE_DIM or x_print.shape[1] != PRINT_DIM:
            raise ValueError(
                f"Expected dimensions ({FACE_DIM}, {VOICE_DIM}, {PRINT_DIM}), "
                f"got ({x_face.shape[1]}, {x_voice.shape[1]}, {x_print.shape[1]})."
            )
        return x_face, x_voice, x_print

    def fit(self, samples: ModalityTuple) -> "EarlyFeatureFusionExtractor":
        """Fit independent StandardScaler on each modality."""
        x_face, x_voice, x_print = self._prepare_modalities(samples)
        self.scaler_face.fit(x_face)
        self.scaler_voice.fit(x_voice)
        self.scaler_print.fit(x_print)
        self.is_fitted_ = True
        return self

    def transform(self, samples: ModalityTuple) -> np.ndarray:
        """Normalize each modality independently and concatenate into a 686D float32 matrix."""
        if not self.is_fitted_:
            raise RuntimeError("Extractor must be fitted before calling transform().")
        x_face, x_voice, x_print = self._prepare_modalities(samples)

        z_face = self.scaler_face.transform(x_face)
        z_voice = self.scaler_voice.transform(x_voice)
        z_print = self.scaler_print.transform(x_print)

        x_fused = np.hstack([z_face, z_voice, z_print]).astype(np.float32)
        assert x_fused.shape[1] == FUSED_DIM
        return x_fused


class CalibratedSVM(BaseBiometricModel):
    """
    Standard RBF Support Vector Classifier with Platt scaling (probability=True).
    Used as the classifier for unimodal baselines and for the 686D fused vectors.
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
            probability=True,          # Platt scaling for calibrated probabilities P(Y=k|x)
            class_weight="balanced",
            random_state=random_state,
        )
        self.classes_ = None

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
        if self.scaler is not None:
            arr = self.scaler.fit_transform(arr) if fit else self.scaler.transform(arr)
        return arr.astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibratedSVM":
        X_scaled = self._scale(X, fit=True)
        self.svc.fit(X_scaled, np.asarray(y, dtype=np.int64))
        self.classes_ = np.asarray(self.svc.classes_, dtype=np.int64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.svc.predict(self._scale(X)), dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.svc.predict_proba(self._scale(X)), dtype=np.float32)


class EarlyFusionMultimodalModel(BaseBiometricModel):
    """
    End-to-end Task 4.2 Multimodal Model:
    Combines `EarlyFeatureFusionExtractor` (3 StandardScalers -> 686D) + `CalibratedSVM`.
    """

    def __init__(self, C: float = 10.0, gamma: str = "scale", random_state: int = 42):
        self.fusion_extractor = EarlyFeatureFusionExtractor()
        self.classifier = CalibratedSVM(C=C, gamma=gamma, use_scaler=False, random_state=random_state)
        self.classes_ = None

    def fit(self, X: ModalityTuple, y: np.ndarray) -> "EarlyFusionMultimodalModel":
        X_fused = self.fusion_extractor.fit_transform(X)
        self.classifier.fit(X_fused, y)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, X: ModalityTuple) -> np.ndarray:
        X_fused = self.fusion_extractor.transform(X)
        return self.classifier.predict(X_fused)

    def predict_proba(self, X: ModalityTuple) -> np.ndarray:
        X_fused = self.fusion_extractor.transform(X)
        return self.classifier.predict_proba(X_fused)
