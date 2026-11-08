"""
src/face_classifiers.py
Task 1.3 Face Model Training, Probability Calibration, & Serialization.

Pipeline (per the task sheet):
  1. Stratified 5-Fold CV on all classes (Class 0 impostors + enrolled/genuine).
  2. RBF-SVM grid search: C in {0.1, 1, 10, 100}, gamma in {'scale','auto',0.01}.
  3. Platt-scaled probabilities P(Y=k|x) (one Platt sigmoid per identity, rows normalised to sum to 1).
  4. Random Forest (150 trees, depth 15, min split 2) with out-of-bag estimation.
  5. Compare HOG+SVM / LBP+SVM / PCA+SVM by Macro F1 and False Acceptance on Class 0.
  6. Serialise extractor + scaler + model to models/face/best_face_model.joblib.
"""
import os
import sys
import itertools

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    sys.modules["src.face_classifiers"] = sys.modules[__name__]

import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from src.face_extractors import HOGExtractor, LBPExtractor, PCAFaceExtractor
from src.base_interfaces import BaseBiometricModel

C_GRID = [0.1, 1.0, 10.0, 100.0]
GAMMA_GRID = ["scale", "auto", 0.01]


class FaceSVMModel(BaseBiometricModel):
    """
    RBF-SVM with Platt scaling.

    One binary RBF-SVC (with its own Platt sigmoid, probability=True) is trained per identity;
    predict_proba normalises the per-identity probabilities so every row sums to 1.0.
    A confident match therefore keeps a high probability instead of being diluted over all
    classes by pairwise coupling, which is what makes the 0.85 / 0.50 thresholds usable.
    """

    def __init__(self, C=10.0, gamma="scale"):
        self.C = C
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model = OneVsRestClassifier(
            SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced",
                probability=True, random_state=42)
        )
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FaceSVMModel":
        self.model.fit(self.scaler.fit_transform(X), y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))


class FaceRFModel(BaseBiometricModel):
    """Random Forest (150 trees, depth 15, min split 2) with out-of-bag estimation."""

    def __init__(self, n_estimators=150, max_depth=15, min_samples_split=2):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
            oob_score=True, n_jobs=-1, random_state=42
        )
        self.scaler = StandardScaler()
        self.classes_ = None
        self.oob_score_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FaceRFModel":
        self.model.fit(self.scaler.fit_transform(X), y)
        self.classes_ = self.model.classes_
        self.oob_score_ = float(self.model.oob_score_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))


def load_dataset(processed_root: str):
    images, labels, label_names = [], [], {}
    for subject in sorted(os.listdir(processed_root)):
        s_dir = os.path.join(processed_root, subject)
        if not os.path.isdir(s_dir):
            continue
        try:
            cid = int(subject.split("_")[1])
        except (IndexError, ValueError):
            continue
        label_names[cid] = subject
        for f in sorted(os.listdir(s_dir)):
            if f.endswith((".png", ".pgm")):
                img = cv2.imread(os.path.join(s_dir, f), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.shape == (100, 100):
                    images.append(img)
                    labels.append(cid)
    print(f"[DATA] Loaded {len(images)} images across {len(set(labels))} classes.")
    return images, np.array(labels), label_names


def make_fold_features(extractor_cls, images, labels, splits):
    """Fit the extractor on each training fold only (no leakage) and cache the features."""
    folds = []
    for tr, va in splits:
        tr_imgs = [images[i] for i in tr]
        va_imgs = [images[i] for i in va]
        ex = extractor_cls().fit(tr_imgs)
        folds.append((ex.transform(tr_imgs), ex.transform(va_imgs)))
    return folds


def far_class0(y_true, y_pred) -> float:
    """False Acceptance on Class 0: share of impostor samples accepted as a genuine identity."""
    mask = y_true == 0
    return float(np.mean(y_pred[mask] != 0)) if mask.any() else 0.0


def grid_search_svm(folds, labels, splits):
    """Grid search C x gamma by Macro F1 (fast multiclass SVC; Platt is applied afterwards)."""
    best = None
    for C, gamma in itertools.product(C_GRID, GAMMA_GRID):
        f1s = []
        for (Xtr, Xva), (tr, va) in zip(folds, splits):
            sc = StandardScaler().fit(Xtr)
            clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced").fit(sc.transform(Xtr), labels[tr])
            f1s.append(f1_score(labels[va], clf.predict(sc.transform(Xva)), average="macro", zero_division=0))
        score = float(np.mean(f1s))
        if best is None or score > best[0]:
            best = (score, C, gamma)
    return best


def cross_validate_model(model_factory, folds, labels, splits):
    """Stratified CV -> (mean Macro F1, False Acceptance on Class 0) from out-of-fold predictions."""
    f1s = []
    oof_true, oof_pred = [], []
    for (Xtr, Xva), (tr, va) in zip(folds, splits):
        m = model_factory().fit(Xtr, labels[tr])
        pred = m.predict(Xva)
        f1s.append(f1_score(labels[va], pred, average="macro", zero_division=0))
        oof_true.append(labels[va])
        oof_pred.append(pred)
    return float(np.mean(f1s)), far_class0(np.concatenate(oof_true), np.concatenate(oof_pred))


def augment_image(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Classical webcam-condition augmentation (no deep learning): random shift / scale / rotation,
    mirror, gamma (brightness), blur and sensor noise, then re-equalise and keep 100x100 uint8.
    """
    h, w = img.shape
    angle = rng.uniform(-12, 12)
    scale = rng.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += rng.uniform(-8, 8, size=2)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.rand() < 0.5:
        out = cv2.flip(out, 1)
    gamma = rng.uniform(0.7, 1.4)
    out = (255.0 * (out / 255.0) ** gamma).astype(np.uint8)
    if rng.rand() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.5, 1.2))
    out = np.clip(out.astype(np.float32) + rng.normal(0, rng.uniform(0, 6), out.shape), 0, 255).astype(np.uint8)
    return cv2.equalizeHist(out)


def augment_dataset(images, labels, copies: int = 4, seed: int = 42):
    """Return originals + `copies` augmented variants of every image (labels repeated)."""
    rng = np.random.RandomState(seed)
    aug_imgs, aug_labels = list(images), list(labels)
    for img, lab in zip(images, labels):
        for _ in range(copies):
            aug_imgs.append(augment_image(img, rng))
            aug_labels.append(lab)
    return aug_imgs, np.array(aug_labels)


def train_and_select_best(processed_root: str = "data/processed", model_output_dir: str = "models/face"):
    os.makedirs(model_output_dir, exist_ok=True)
    images, labels, label_names = load_dataset(processed_root)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(images, labels))

    extractors = [("HOG", HOGExtractor), ("LBP", LBPExtractor), ("PCA", PCAFaceExtractor)]
    results = {}
    print("\n=== Stratified 5-Fold CV + RBF-SVM grid search (C x gamma) ===")
    for name, ex_cls in extractors:
        folds = make_fold_features(ex_cls, images, labels, splits)
        _, C, gamma = grid_search_svm(folds, labels, splits)
        f1, far = cross_validate_model(lambda C=C, g=gamma: FaceSVMModel(C, g), folds, labels, splits)
        results[name] = {"ex_cls": ex_cls, "C": C, "gamma": gamma, "f1": f1, "far": far, "folds": folds}
        print(f"  {name}+SVM  best C={C:<6} gamma={str(gamma):<6} | Macro F1={f1:.4f} | FAR(Class 0)={far:.4f}")

    rf_folds = results["HOG"]["folds"]
    rf_f1, rf_far = cross_validate_model(FaceRFModel, rf_folds, labels, splits)
    print(f"  HOG+RF   (150 trees, depth 15)        | Macro F1={rf_f1:.4f} | FAR(Class 0)={rf_far:.4f}")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["f1"] - kv[1]["far"], reverse=True)
    print(f"\n[RANKING by Macro F1 - FAR] {[k for k, _ in ranked]}")

    # Task 1.4 pipeline is specified on the HOG extractor, so the deployed artifact is HOG+SVM.
    sel = results["HOG"]
    print(f"[SELECTED] HOG+SVM (C={sel['C']}, gamma={sel['gamma']}) for the live webcam pipeline.")

    best_ex = HOGExtractor().fit(images)
    X_all = best_ex.transform(images)
    rf = FaceRFModel().fit(X_all, labels)
    print(f"[RF] Out-of-bag score = {rf.oob_score_:.4f}")
    joblib.dump({"extractor": best_ex, "model": rf, "label_names": label_names},
                os.path.join(model_output_dir, "face_rf_model.joblib"))

    # Deployed SVM is trained on originals + augmented webcam-like variants for cross-camera robustness.
    aug_images, aug_labels = augment_dataset(images, labels, copies=4)
    print(f"[AUGMENT] Final SVM trained on {len(aug_images)} images ({len(images)} originals x5).")
    X_aug = best_ex.transform(aug_images)
    best_model = FaceSVMModel(sel["C"], sel["gamma"]).fit(X_aug, aug_labels)

    proba = best_model.predict_proba(X_all)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), "Probabilities must sum to 1.0"
    print(f"[CHECK] Probabilities sum to 1.0 across {proba.shape[1]} identities (incl. Class 0).")

    artifact = {
        "extractor": best_ex,
        "model": best_model,
        "label_names": label_names,
        "best_config": "HOG+SVM",
        "best_params": {"C": sel["C"], "gamma": sel["gamma"]},
        "best_f1": sel["f1"],
        "far_class0": sel["far"],
        "classes": best_model.classes_.tolist(),
    }
    out_path = os.path.join(model_output_dir, "best_face_model.joblib")
    joblib.dump(artifact, out_path)
    print(f"[SAVED] Model artifact -> {out_path}")
    return artifact


if __name__ == "__main__":
    train_and_select_best()
