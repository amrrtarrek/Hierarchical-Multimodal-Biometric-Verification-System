"""
src/fingerprint_classifiers.py
Task 3.2 (Steps 4-5): Standalone Fingerprint SVM Model (RBF Kernel + Platt Calibration).

Implements:
1. FingerprintSVMModel (BaseBiometricModel):
   Dedicated Support Vector Classifier (RBF kernel, Platt probability calibration)
   with StandardScaler on 64D Gabor filter bank feature vectors.
   Supports both direct BaseBiometricModel calls (`.predict_proba(X)`) and
   dict-style lookup (`artifact["model"]`, `artifact["extractor"]`).
2. Leakage-safe Stratified 5-Fold Cross-Validation with grid search over
   C in {0.1, 1.0, 10.0, 100.0} and gamma in {'scale', 'auto', 0.01}.
3. Serialization of the fitted extractor and classifier artifact to
   `models/fingerprint/best_fingerprint_model.joblib` and feature matrices
   `X_print.npy`, `y_print.npy` to `data/processed/fingerprint/`.
"""
from __future__ import annotations

import itertools
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.base_interfaces import BaseBiometricModel
from src.fingerprint_extractors import GaborFingerprintExtractor
from src.fingerprint_preprocessing import augment_fingerprint, load_fingerprint_dataset

C_GRID: List[float] = [0.1, 1.0, 10.0, 100.0]
GAMMA_GRID: List[Any] = ["scale", "auto", 0.01]

DEFAULT_PRINT_LABELS: Dict[int, str] = {
    0: "Class 0 (Impostor)",
    1: "Member 1 (Admin)",
    2: "Member 2 (Operator)",
    3: "Member 3 (Auditor)",
    4: "Member 4 (Auditor)",
}


class _DualFingerprintArtifactMixin:
    """
    Allows the serialized FingerprintSVMModel to act both as a BaseBiometricModel
    (`.predict_proba(X)`, `.predict(X)`, `.classes_`) and as a dictionary
    (`artifact["model"]`, `artifact["extractor"]`, `artifact["binarize"]`).
    """

    def _artifact_dict(self) -> Dict[str, Any]:
        return {
            "model": self,
            "scaler": getattr(self, "scaler", None),
            "extractor": getattr(self, "extractor", None),
            "binarize": getattr(self, "binarize", True),
            "label_names": getattr(self, "label_names", DEFAULT_PRINT_LABELS),
            "best_config": getattr(self, "best_config", "Gabor64D+RBF-SVM"),
            "best_params": getattr(self, "best_params", {"C": self.C, "gamma": self.gamma}),
            "best_f1": getattr(self, "best_f1", None),
            "far_class0": getattr(self, "far_class0", None),
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


class FingerprintSVMModel(BaseBiometricModel, _DualFingerprintArtifactMixin):
    """
    Task 3.2 Step 4: Dedicated Fingerprint SVM (RBF kernel, Platt probability calibration)
    on 64D Gabor filter bank feature vectors.
    """

    def __init__(
        self,
        C: float = 10.0,
        gamma: str | float = "scale",
        random_state: int = 42,
    ) -> None:
        self.C = float(C)
        self.gamma = gamma
        self.random_state = int(random_state)

        self.scaler = StandardScaler()
        base_svc = SVC(
            kernel="rbf",
            C=self.C,
            gamma=self.gamma,
            class_weight="balanced",
            probability=True,  # Platt probability calibration
            random_state=self.random_state,
        )
        self.model = OneVsRestClassifier(base_svc)
        self.classes_: Optional[np.ndarray] = None
        self.extractor: Optional[GaborFingerprintExtractor] = None
        self.binarize: bool = True
        self.label_names: Dict[int, str] = dict(DEFAULT_PRINT_LABELS)
        self.best_config: str = "Gabor64D+RBF-SVM"
        self.best_params: Dict[str, Any] = {"C": self.C, "gamma": self.gamma}
        self.best_f1: Optional[float] = None
        self.far_class0: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FingerprintSVMModel":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        X_scaled = self.scaler.fit_transform(X_arr)
        self.model.fit(X_scaled, y_arr)
        self.classes_ = np.asarray(self.model.classes_, dtype=np.int64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        top_idx = np.argmax(probs, axis=1)
        return self.classes_[top_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
        X_scaled = self.scaler.transform(X_arr)
        raw_probs = np.asarray(self.model.predict_proba(X_scaled), dtype=np.float64)
        row_sums = np.sum(raw_probs, axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
        return (raw_probs / row_sums).astype(np.float32)


# Alias for backward compatibility with Member 3's original class name
FingerprintModel = FingerprintSVMModel


def augment_training_fold_images(
    images: List[np.ndarray],
    labels: np.ndarray,
    copies: int = 3,
    seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Augment training images only (after train/val split to prevent data leakage).
    """
    rng = np.random.default_rng(seed)
    out_imgs = list(images)
    out_labels = list(labels)
    for img, lbl in zip(images, labels):
        for _ in range(copies):
            out_imgs.append(augment_fingerprint(img, rng))
            out_labels.append(int(lbl))
    return out_imgs, np.asarray(out_labels, dtype=np.int64)


def compute_print_far_frr(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    imp_mask = y_t == 0
    far = float(np.mean(y_p[imp_mask] != 0)) if np.any(imp_mask) else 0.0
    gen_mask = y_t != 0
    frr = float(np.mean(y_p[gen_mask] == 0)) if np.any(gen_mask) else 0.0
    return far, frr


def train_and_select_best_fingerprint_model(
    processed_print_root: str = "data/processed/fingerprint",
    model_output_dir: str = "models/fingerprint",
) -> Dict[str, Any]:
    """
    Execute Task 3.2:
    1. Load preprocessed 128x128 fingerprint dataset (Class 0 Impostors + Enrolled Members 1..4).
    2. Extract 64D Gabor filter bank features and save `X_print.npy` and `y_print.npy`.
    3. Run Stratified 5-Fold Cross-Validation over C and gamma.
    4. Train final calibrated RBF-SVM and serialize to `models/fingerprint/best_fingerprint_model.joblib`.
    """
    os.makedirs(model_output_dir, exist_ok=True)
    images, y_print, _ = load_fingerprint_dataset(processed_print_root, include_benchmark=False)
    if len(images) == 0:
        raise RuntimeError(f"No processed fingerprints found in '{processed_print_root}'. Run process_fingerprint_data.py first.")

    extractor = GaborFingerprintExtractor().fit(images)
    X_print = extractor.transform(images)

    # Save base 64D feature matrix (40, 64) and labels
    np.save(os.path.join(processed_print_root, "X_print.npy"), X_print)
    np.save(os.path.join(processed_print_root, "y_print.npy"), y_print)

    print("=" * 72)
    print("  TASK 3.2: FINGERPRINT 2D GABOR FILTER BANK EXTRACTION & SVM MODEL")
    print("=" * 72)
    print(f"[DATA] Extracted X_print={X_print.shape} (dtype={X_print.dtype}) across classes {sorted(set(y_print.tolist()))}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(images, y_print))

    best_entry: Optional[Dict[str, Any]] = None
    for C, gamma in itertools.product(C_GRID, GAMMA_GRID):
        fold_f1s = []
        oof_true, oof_pred = [], []
        for fold_i, (tr_idx, va_idx) in enumerate(splits):
            tr_imgs = [images[i] for i in tr_idx]
            va_imgs = [images[i] for i in va_idx]
            aug_tr_imgs, aug_tr_y = augment_training_fold_images(tr_imgs, y_print[tr_idx], copies=3, seed=42 + fold_i)
            X_tr = extractor.transform(aug_tr_imgs)
            X_va = extractor.transform(va_imgs)

            clf = FingerprintSVMModel(C=C, gamma=gamma).fit(X_tr, aug_tr_y)
            preds = clf.predict(X_va)
            fold_f1s.append(f1_score(y_print[va_idx], preds, average="macro", zero_division=0))
            oof_true.append(y_print[va_idx])
            oof_pred.append(preds)

        macro_f1 = float(np.mean(fold_f1s))
        far_c0, frr_gen = compute_print_far_frr(np.concatenate(oof_true), np.concatenate(oof_pred))
        score = macro_f1 - 0.3 * far_c0
        entry = {"C": C, "gamma": gamma, "macro_f1": macro_f1, "far_class0": far_c0, "frr": frr_gen, "score": score}
        if best_entry is None or score > best_entry["score"]:
            best_entry = entry

    assert best_entry is not None
    print(
        f"  [5-Fold CV Best] RBF-SVM C={best_entry['C']:<6} gamma={str(best_entry['gamma']):<6} | "
        f"Macro F1={best_entry['macro_f1']:.4f} | FAR(Class 0)={best_entry['far_class0']:.4f} | FRR={best_entry['frr']:.4f}"
    )

    # Train final deployed model on augmented full dataset
    aug_imgs, aug_y = augment_training_fold_images(images, y_print, copies=4, seed=42)
    X_aug = extractor.transform(aug_imgs)

    best_model = FingerprintSVMModel(C=best_entry["C"], gamma=best_entry["gamma"]).fit(X_aug, aug_y)
    best_model.extractor = extractor
    best_model.binarize = True
    best_model.label_names = dict(DEFAULT_PRINT_LABELS)
    best_model.best_params = {"C": best_entry["C"], "gamma": best_entry["gamma"]}
    best_model.best_f1 = best_entry["macro_f1"]
    best_model.far_class0 = best_entry["far_class0"]

    proba = best_model.predict_proba(X_print)
    preds = best_model.predict(X_print)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    acc = float(accuracy_score(y_print, preds))
    far_full, frr_full = compute_print_far_frr(y_print, preds)

    out_path = os.path.join(model_output_dir, "best_fingerprint_model.joblib")
    joblib.dump(best_model, out_path)

    print("-" * 72)
    print(f"  [SAVED]  Model artifact -> '{out_path}'")
    print(f"  [SAVED]  Feature matrix -> '{processed_print_root}/X_print.npy' {X_print.shape}")
    print(f"  [VERIFY] Calibrated probabilities shape={proba.shape}, row sums=1.0")
    print(f"  [VERIFY] Accuracy={acc * 100:.2f}%, FAR(Class 0)={far_full:.4f}, FRR={frr_full:.4f}")
    print("=" * 72)

    return {
        "model": best_model,
        "extractor": extractor,
        "X_print": X_print,
        "y_print": y_print,
        "cv_best": best_entry,
        "artifact_path": out_path,
    }


_DualFingerprintArtifactMixin.__module__ = "src.fingerprint_classifiers"
FingerprintSVMModel.__module__ = "src.fingerprint_classifiers"


if __name__ == "__main__":
    from src.fingerprint_classifiers import train_and_select_best_fingerprint_model as _run_train
    _run_train()

