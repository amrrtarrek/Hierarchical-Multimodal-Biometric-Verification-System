"""
src/voice_classifiers.py
Task 2.3: Speaker Verification Models (SVM & Logistic Regression) (Day 3 | Core).

Implements:
1. SpeakerSVMModel (BaseBiometricModel):
   Support Vector Classifier (linear and polynomial kernels) with Platt probability
   calibration (probability=True) and StandardScaler on 46D acoustic vectors.
2. SpeakerLogisticRegressionModel (BaseBiometricModel):
   One-vs-Rest (OvR) Logistic Regression with L2 penalty as a comparative baseline.
3. Stratified 5-Fold Cross-Validation across all enrolled speaker classes (1..4)
   plus Class 0 (Impostors).
4. Hyperparameter tuning of regularizer C, tolerance tol, and kernel parameters
   to maximize speaker separation under ambient noise.
5. Serialization of the fitted speaker classifier artifact to
   `models/voice/best_voice_model.joblib` (supporting both direct BaseBiometricModel
   methods `.predict_proba(X)` and dict-style `artifact["model"]` lookup).
"""
from __future__ import annotations

import itertools
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    sys.modules["src.voice_classifiers"] = sys.modules[__name__]

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.base_interfaces import BaseBiometricModel
from src.voice_extractors import VoiceFeatureExtractor, load_processed_voice_dataset
from src.voice_preprocessing import SAMPLE_RATE, condition_voice_waveform

C_GRID: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
TOL_GRID: List[float] = [1e-4, 1e-3, 1e-2]
SVM_KERNEL_GRID: List[Tuple[str, int]] = [
    ("linear", 1),
    ("poly", 2),
    ("poly", 3),
]

DEFAULT_SPEAKER_NAMES: Dict[int, str] = {
    0: "Class 0 (Impostor)",
    1: "Member 1 (Admin)",
    2: "Member 2 (Operator)",
    3: "Member 3 (Auditor)",
    4: "Member 4 (Auditor)",
}


class _DualArtifactMixin:
    """
    Mixin enabling a serialized BaseBiometricModel instance to also behave like
    an artifact dictionary (`artifact["model"]`, `artifact["extractor"]`, etc.)
    so both `model.predict_proba(X)` (Member 4) and `artifact["model"]` (Member 1/3)
    work seamlessly.
    """

    def _artifact_dict(self) -> Dict[str, Any]:
        return {
            "model": self,
            "scaler": getattr(self, "scaler", None),
            "extractor": getattr(self, "extractor", None),
            "label_names": getattr(self, "label_names", DEFAULT_SPEAKER_NAMES),
            "best_config": getattr(self, "best_config", self.__class__.__name__),
            "best_params": getattr(self, "best_params", {}),
            "best_f1": getattr(self, "best_f1", None),
            "far_class0": getattr(self, "far_class0", None),
            "cv_report": getattr(self, "cv_report", {}),
            "classes": self.classes_.tolist() if getattr(self, "classes_", None) is not None else [],
        }

    def __getitem__(self, key: str) -> Any:
        d = self._artifact_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._artifact_dict().get(key, default)

    def keys(self):
        return self._artifact_dict().keys()

    def __contains__(self, key: str) -> bool:
        return key in self._artifact_dict()


class SpeakerSVMModel(BaseBiometricModel, _DualArtifactMixin):
    """
    Step 1: Support Vector Classifier (linear and polynomial kernels) with Platt
    probability calibration on 46D acoustic feature vectors.
    """

    def __init__(
        self,
        kernel: str = "linear",
        C: float = 1.0,
        tol: float = 1e-3,
        degree: int = 2,
        gamma: str | float = "scale",
        coef0: float = 1.0,
        random_state: int = 42,
    ) -> None:
        if kernel not in ("linear", "poly", "rbf"):
            raise ValueError(f"Unsupported SVM kernel '{kernel}'. Expected 'linear' or 'poly'.")
        self.kernel = kernel
        self.C = float(C)
        self.tol = float(tol)
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.random_state = int(random_state)

        self.scaler = StandardScaler()
        base_svc = SVC(
            kernel=self.kernel,
            C=self.C,
            tol=self.tol,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            class_weight="balanced",
            probability=True,  # Enables Platt scaling probability calibration
            random_state=self.random_state,
        )
        self.model = OneVsRestClassifier(base_svc)
        self.classes_: Optional[np.ndarray] = None
        self.extractor: Optional[VoiceFeatureExtractor] = None
        self.label_names: Dict[int, str] = dict(DEFAULT_SPEAKER_NAMES)
        self.best_config: str = f"SVM({self.kernel})"
        self.best_params: Dict[str, Any] = {
            "kernel": self.kernel,
            "C": self.C,
            "tol": self.tol,
            "degree": self.degree,
        }
        self.best_f1: Optional[float] = None
        self.far_class0: Optional[float] = None
        self.cv_report: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpeakerSVMModel":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        X_scaled = self.scaler.fit_transform(X_arr)
        self.model.fit(X_scaled, y_arr)
        self.classes_ = np.asarray(self.model.classes_, dtype=np.int64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        top_indices = np.argmax(probs, axis=1)
        return self.classes_[top_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
        X_scaled = self.scaler.transform(X_arr)
        raw_probs = np.asarray(self.model.predict_proba(X_scaled), dtype=np.float64)
        row_sums = np.sum(raw_probs, axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
        calibrated = (raw_probs / row_sums).astype(np.float32)
        return calibrated


class SpeakerLogisticRegressionModel(BaseBiometricModel, _DualArtifactMixin):
    """
    Step 2: Multinomial / One-vs-Rest Logistic Regression model with L2 penalty
    as a comparative baseline on 46D acoustic feature vectors.
    """

    def __init__(
        self,
        C: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.C = float(C)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

        self.scaler = StandardScaler()
        base_lr = LogisticRegression(
            C=self.C,
            tol=self.tol,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model = OneVsRestClassifier(base_lr)
        self.classes_: Optional[np.ndarray] = None
        self.extractor: Optional[VoiceFeatureExtractor] = None
        self.label_names: Dict[int, str] = dict(DEFAULT_SPEAKER_NAMES)
        self.best_config: str = "LogisticRegression(OvR-L2)"
        self.best_params: Dict[str, Any] = {"C": self.C, "tol": self.tol, "penalty": "l2"}
        self.best_f1: Optional[float] = None
        self.far_class0: Optional[float] = None
        self.cv_report: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpeakerLogisticRegressionModel":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        X_scaled = self.scaler.fit_transform(X_arr)
        self.model.fit(X_scaled, y_arr)
        self.classes_ = np.asarray(self.model.classes_, dtype=np.int64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        top_indices = np.argmax(probs, axis=1)
        return self.classes_[top_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
        X_scaled = self.scaler.transform(X_arr)
        raw_probs = np.asarray(self.model.predict_proba(X_scaled), dtype=np.float64)
        row_sums = np.sum(raw_probs, axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
        return (raw_probs / row_sums).astype(np.float32)


def compute_far_frr(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute False Acceptance Rate on Class 0 (impostors accepted as genuine)
    and False Rejection Rate on genuine speakers (rejected as Class 0).
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    imp_mask = y_t == 0
    far = float(np.mean(y_p[imp_mask] != 0)) if np.any(imp_mask) else 0.0

    gen_mask = y_t != 0
    frr = float(np.mean(y_p[gen_mask] == 0)) if np.any(gen_mask) else 0.0
    return far, frr


def augment_voice_features_with_noise(
    X: np.ndarray,
    y: np.ndarray,
    copies: int = 3,
    noise_levels: Tuple[float, ...] = (0.02, 0.05, 0.08),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 4: Simulate ambient microphone noise and slight vocal/channel gain drift
    in feature space (scaled per-feature std) to improve speaker separation under ambient noise.
    """
    rng = np.random.default_rng(seed)
    X_orig = np.asarray(X, dtype=np.float32)
    y_orig = np.asarray(y, dtype=np.int64)

    feat_std = np.std(X_orig, axis=0, keepdims=True)
    feat_std = np.where(feat_std > 1e-8, feat_std, 1.0)

    X_list = [X_orig]
    y_list = [y_orig]

    for c in range(copies):
        sigma = noise_levels[c % len(noise_levels)]
        noise = rng.normal(0.0, sigma, size=X_orig.shape).astype(np.float32) * feat_std
        gain = rng.uniform(0.96, 1.04, size=(X_orig.shape[0], 1)).astype(np.float32)
        X_noisy = (X_orig * gain + noise).astype(np.float32)
        X_list.append(X_noisy)
        y_list.append(y_orig.copy())

    return np.vstack(X_list).astype(np.float32), np.concatenate(y_list).astype(np.int64)


def augment_waveform_with_ambient_noise(
    audio: np.ndarray,
    snr_db: float = 20.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Inject colored ambient room noise (low-frequency HVAC hum + pink/white floor)
    into a 1D waveform at a specified SNR (dB) and re-condition.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = len(x)
    t = np.arange(n, dtype=np.float32) / float(SAMPLE_RATE)

    sig_power = float(np.mean(x ** 2))
    if sig_power <= 1e-10:
        return x.copy()

    white = rng.standard_normal(n).astype(np.float32)
    hvac_hum = (0.4 * np.sin(2.0 * np.pi * 60.0 * t) + 0.2 * np.sin(2.0 * np.pi * 120.0 * t)).astype(np.float32)
    ambient = white + hvac_hum
    ambient_power = float(np.mean(ambient ** 2))

    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / max(ambient_power, 1e-12))
    mixed = x + (ambient * scale).astype(np.float32)
    return condition_voice_waveform(mixed)


def evaluate_model_cv(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    augment_train: bool = True,
    val_noise_std: float = 0.03,
) -> Dict[str, float]:
    """
    Evaluate a speaker model using Stratified 5-Fold Cross-Validation on both clean
    and ambient-noise-perturbed validation folds.
    """
    f1_clean_list, acc_clean_list = [], []
    f1_noisy_list, acc_noisy_list = [], []
    oof_true, oof_pred_clean, oof_pred_noisy = [], [], []

    feat_std = np.std(X, axis=0, keepdims=True)
    feat_std = np.where(feat_std > 1e-8, feat_std, 1.0)
    rng = np.random.default_rng(777)

    for tr_idx, va_idx in splits:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        if augment_train:
            X_tr_fit, y_tr_fit = augment_voice_features_with_noise(X_tr, y_tr, copies=3, seed=42)
        else:
            X_tr_fit, y_tr_fit = X_tr, y_tr

        clf = model_factory().fit(X_tr_fit, y_tr_fit)

        # Clean validation fold
        pred_clean = clf.predict(X_va)
        f1_clean_list.append(f1_score(y_va, pred_clean, average="macro", zero_division=0))
        acc_clean_list.append(accuracy_score(y_va, pred_clean))

        # Ambient noise validation fold
        X_va_noisy = (X_va + rng.normal(0.0, val_noise_std, size=X_va.shape).astype(np.float32) * feat_std).astype(np.float32)
        pred_noisy = clf.predict(X_va_noisy)
        f1_noisy_list.append(f1_score(y_va, pred_noisy, average="macro", zero_division=0))
        acc_noisy_list.append(accuracy_score(y_va, pred_noisy))

        oof_true.append(y_va)
        oof_pred_clean.append(pred_clean)
        oof_pred_noisy.append(pred_noisy)

    y_all = np.concatenate(oof_true)
    p_clean = np.concatenate(oof_pred_clean)
    p_noisy = np.concatenate(oof_pred_noisy)

    far_clean, frr_clean = compute_far_frr(y_all, p_clean)
    far_noisy, frr_noisy = compute_far_frr(y_all, p_noisy)

    return {
        "macro_f1_clean": float(np.mean(f1_clean_list)),
        "accuracy_clean": float(np.mean(acc_clean_list)),
        "far_class0_clean": far_clean,
        "frr_clean": frr_clean,
        "macro_f1_noisy": float(np.mean(f1_noisy_list)),
        "accuracy_noisy": float(np.mean(acc_noisy_list)),
        "far_class0_noisy": far_noisy,
        "frr_noisy": frr_noisy,
    }


def train_and_select_best_voice_model(
    processed_voice_root: str = "data/processed/voice",
    model_output_dir: str = "models/voice",
) -> Dict[str, Any]:
    """
    Execute full Task 2.3 training, 5-fold cross-validation, hyperparameter tuning,
    and model artifact serialization to `models/voice/best_voice_model.joblib`.
    """
    os.makedirs(model_output_dir, exist_ok=True)

    x_path = os.path.join(processed_voice_root, "X_voice.npy")
    y_path = os.path.join(processed_voice_root, "y_speaker.npy")
    if os.path.isfile(x_path) and os.path.isfile(y_path):
        X_voice = np.load(x_path)
        y_speaker = np.load(y_path)
    else:
        X_voice, y_speaker, _, _ = load_processed_voice_dataset(processed_voice_root)

    print("=" * 72)
    print("  TASK 2.3: SPEAKER VERIFICATION MODELS (SVM & LOGISTIC REGRESSION)")
    print("=" * 72)
    print(f"[DATA] Loaded X_voice={X_voice.shape} across classes {sorted(set(y_speaker.tolist()))}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X_voice, y_speaker))

    # 1. Grid Search SVM (Linear & Polynomial kernels) over C and tol
    best_svm_entry: Optional[Dict[str, Any]] = None
    svm_Search_log: List[Dict[str, Any]] = []

    for (kernel, degree), C, tol in itertools.product(SVM_KERNEL_GRID, C_GRID, TOL_GRID):
        metrics = evaluate_model_cv(
            lambda k=kernel, d=degree, c=C, t=tol: SpeakerSVMModel(kernel=k, C=c, tol=t, degree=d),
            X_voice,
            y_speaker,
            splits,
            augment_train=True,
        )
        # Composite objective: maximize clean & noisy Macro F1 while penalizing Class 0 FAR
        composite_score = (
            0.55 * metrics["macro_f1_clean"]
            + 0.45 * metrics["macro_f1_noisy"]
            - 0.25 * metrics["far_class0_clean"]
            - 0.20 * metrics["far_class0_noisy"]
        )
        entry = {
            "model_type": "SVM",
            "kernel": kernel,
            "degree": degree,
            "C": C,
            "tol": tol,
            "metrics": metrics,
            "score": composite_score,
        }
        svm_Search_log.append(entry)
        if best_svm_entry is None or composite_score > best_svm_entry["score"]:
            best_svm_entry = entry

    assert best_svm_entry is not None
    sm = best_svm_entry["metrics"]
    k_desc = (
        f"poly(deg={best_svm_entry['degree']})"
        if best_svm_entry["kernel"] == "poly"
        else "linear"
    )
    print(
        f"  [SVM Best] kernel={k_desc:<12} C={best_svm_entry['C']:<6} tol={best_svm_entry['tol']:<7} | "
        f"Macro F1 (clean)={sm['macro_f1_clean']:.4f} | Macro F1 (noisy)={sm['macro_f1_noisy']:.4f} | "
        f"FAR(Class 0)={sm['far_class0_clean']:.4f}"
    )

    # 2. Grid Search Multinomial / OvR Logistic Regression (L2 Penalty) over C and tol
    best_lr_entry: Optional[Dict[str, Any]] = None
    for C, tol in itertools.product(C_GRID, TOL_GRID):
        metrics = evaluate_model_cv(
            lambda c=C, t=tol: SpeakerLogisticRegressionModel(C=c, tol=t),
            X_voice,
            y_speaker,
            splits,
            augment_train=True,
        )
        composite_score = (
            0.55 * metrics["macro_f1_clean"]
            + 0.45 * metrics["macro_f1_noisy"]
            - 0.25 * metrics["far_class0_clean"]
            - 0.20 * metrics["far_class0_noisy"]
        )
        entry = {
            "model_type": "LogisticRegression",
            "C": C,
            "tol": tol,
            "metrics": metrics,
            "score": composite_score,
        }
        if best_lr_entry is None or composite_score > best_lr_entry["score"]:
            best_lr_entry = entry

    assert best_lr_entry is not None
    lm = best_lr_entry["metrics"]
    print(
        f"  [LR  Best] penalty=l2 (OvR) C={best_lr_entry['C']:<6} tol={best_lr_entry['tol']:<7} | "
        f"Macro F1 (clean)={lm['macro_f1_clean']:.4f} | Macro F1 (noisy)={lm['macro_f1_noisy']:.4f} | "
        f"FAR(Class 0)={lm['far_class0_clean']:.4f}"
    )

    # 3. Fit final models on full ambient-noise-augmented dataset
    extractor = VoiceFeatureExtractor().fit([])
    X_aug, y_aug = augment_voice_features_with_noise(X_voice, y_speaker, copies=4, seed=42)

    final_svm = SpeakerSVMModel(
        kernel=best_svm_entry["kernel"],
        C=best_svm_entry["C"],
        tol=best_svm_entry["tol"],
        degree=best_svm_entry["degree"],
    ).fit(X_aug, y_aug)
    final_svm.extractor = extractor
    final_svm.label_names = dict(DEFAULT_SPEAKER_NAMES)
    final_svm.best_config = f"SVM({k_desc})"
    final_svm.best_params = {
        "kernel": best_svm_entry["kernel"],
        "degree": best_svm_entry["degree"],
        "C": best_svm_entry["C"],
        "tol": best_svm_entry["tol"],
    }
    final_svm.best_f1 = sm["macro_f1_clean"]
    final_svm.far_class0 = sm["far_class0_clean"]
    final_svm.cv_report = {"svm_best": best_svm_entry, "lr_best": best_lr_entry}

    final_lr = SpeakerLogisticRegressionModel(
        C=best_lr_entry["C"],
        tol=best_lr_entry["tol"],
    ).fit(X_aug, y_aug)
    final_lr.extractor = extractor
    final_lr.label_names = dict(DEFAULT_SPEAKER_NAMES)
    final_lr.best_f1 = lm["macro_f1_clean"]
    final_lr.far_class0 = lm["far_class0_clean"]
    final_lr.cv_report = {"svm_best": best_svm_entry, "lr_best": best_lr_entry}

    # Select best overall model (or SVM if tied/superior)
    best_model = final_svm if best_svm_entry["score"] >= best_lr_entry["score"] else final_lr

    # Verify Definition of Done on full dataset
    proba = best_model.predict_proba(X_voice)
    preds = best_model.predict(X_voice)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Calibrated probabilities must sum to 1.0!"
    train_acc = float(accuracy_score(y_speaker, preds))
    far_full, frr_full = compute_far_frr(y_speaker, preds)

    best_out_path = os.path.join(model_output_dir, "best_voice_model.joblib")
    lr_out_path = os.path.join(model_output_dir, "voice_logreg_model.joblib")
    joblib.dump(best_model, best_out_path)
    joblib.dump(final_lr, lr_out_path)

    print("-" * 72)
    print(f"  [SELECTED] {best_model.best_config} -> '{best_out_path}'")
    print(f"  [BASELINE] {final_lr.best_config} -> '{lr_out_path}'")
    print(f"  [VERIFY]   Calibrated probabilities shape={proba.shape}, row sums=1.0")
    print(f"  [VERIFY]   Full-set Accuracy={train_acc * 100:.2f}%, FAR(Class 0)={far_full:.4f}, FRR={frr_full:.4f}")
    print("=" * 72)

    return {
        "best_model": best_model,
        "svm_model": final_svm,
        "lr_model": final_lr,
        "svm_cv": best_svm_entry,
        "lr_cv": best_lr_entry,
        "artifact_path": best_out_path,
    }


_DualArtifactMixin.__module__ = "src.voice_classifiers"
SpeakerSVMModel.__module__ = "src.voice_classifiers"
SpeakerLogisticRegressionModel.__module__ = "src.voice_classifiers"


if __name__ == "__main__":
    from src.voice_classifiers import train_and_select_best_voice_model as _run_train
    _run_train()

