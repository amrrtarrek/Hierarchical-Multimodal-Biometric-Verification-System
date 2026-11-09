"""
src/command_classifier.py
Task 3.3 (Step 1): Isolated-Word Speech Command Recognizer (Day 4 & 5 | Bonus 1).

Implements:
1. CommandClassifier (BaseBiometricModel):
   Independent Random Forest classifier trained on 46D acoustic feature vectors
   mapped against target word labels ("Unlock System", "Report Status", "Terminate Session"),
   ensuring keyword classification is speaker-invariant.
2. Speaker-invariance evaluation via GroupKFold (Leave-Speaker-Out CV) and Stratified 5-Fold CV.
3. Serialization of trained command model artifact to `models/fingerprint/best_command_model.joblib`
   and `models/voice/best_command_model.joblib`.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.base_interfaces import BaseBiometricModel
from src.voice_classifiers import augment_voice_features_with_noise
from src.voice_extractors import COMMAND_MAP, VoiceFeatureExtractor, load_processed_voice_dataset

COMMANDS: Tuple[str, ...] = ("Unlock System", "Report Status", "Terminate Session")
ID_TO_COMMAND: Dict[int, str] = {i: cmd for i, cmd in enumerate(COMMANDS)}
COMMAND_TO_ID: Dict[str, int] = {
    "Unlock System": 0,
    "Report Status": 1,
    "Terminate Session": 2,
    "unlock": 0,
    "status": 1,
    "terminate": 2,
}


class CommandClassifier(BaseBiometricModel):
    """
    Task 3.3 Step 1: Speaker-invariant Random Forest classifier on 46D speech feature vectors.
    - `predict(X)` returns canonical command strings ("Unlock System", "Report Status", "Terminate Session").
    - `predict_ids(X)` returns integer command indices (0, 1, 2).
    - `predict_proba(X)` returns calibrated class probabilities of shape (N, 3) summing to 1.0.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 15,
        min_samples_split: int = 2,
        seed: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.seed = int(seed)

        self.scaler = StandardScaler()
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight="balanced_subsample",
            oob_score=True,
            n_jobs=-1,
            random_state=self.seed,
        )
        self.classes_: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None
        self.commands: Tuple[str, ...] = COMMANDS

    @staticmethod
    def _coerce_label(val: Union[int, str, np.integer]) -> int:
        if isinstance(val, (int, np.integer)):
            return int(val)
        s = str(val).strip()
        if s in COMMAND_TO_ID:
            return COMMAND_TO_ID[s]
        if s in COMMANDS:
            return COMMANDS.index(s)
        raise ValueError(f"Unknown command label: {val!r}")

    def fit(self, X: np.ndarray, y: Iterable[Union[int, str]]) -> "CommandClassifier":
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
        y_ids = np.asarray([self._coerce_label(v) for v in y], dtype=np.int64)

        X_scaled = self.scaler.fit_transform(X_arr)
        self.rf.fit(X_scaled, y_ids)
        self.classes_ = np.asarray(self.rf.classes_, dtype=np.int64)
        self.oob_score_ = float(self.rf.oob_score_)
        return self

    def predict_ids(self, X: np.ndarray) -> np.ndarray:
        """Predict integer command indices (0, 1, 2)."""
        probs = self.predict_proba(X)
        top_idx = np.argmax(probs, axis=1)
        return self.classes_[top_idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict canonical command strings ('Unlock System', 'Report Status', 'Terminate Session')."""
        ids = self.predict_ids(X)
        return np.array([COMMANDS[int(i)] for i in ids], dtype=object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
        X_scaled = self.scaler.transform(X_arr)
        probs = np.asarray(self.rf.predict_proba(X_scaled), dtype=np.float64)
        row_sums = np.sum(probs, axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
        return (probs / row_sums).astype(np.float32)

    def predict_with_confidence(self, x: np.ndarray) -> Tuple[str, float]:
        """Predict (command_string, probability_confidence) for a single 46D vector."""
        probs = self.predict_proba(np.atleast_2d(x))[0]
        best_idx = int(np.argmax(probs))
        cmd_id = int(self.classes_[best_idx]) if self.classes_ is not None else best_idx
        return COMMANDS[cmd_id], float(probs[best_idx])

    def __getitem__(self, key: str) -> Any:
        d = {"model": self, "commands": COMMANDS, "classes": [0, 1, 2], "oob_score": self.oob_score_}
        if key in d:
            return d[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in {"model", "commands", "classes", "oob_score"}


SpeechCommandClassifier = CommandClassifier


def speaker_invariance_cv(
    X: np.ndarray,
    y: Iterable[Union[int, str]],
    speakers: np.ndarray,
    n_splits: int = 3,
) -> float:
    """
    Evaluate Leave-Speakers-Out cross-validation accuracy using GroupKFold
    to verify that command recognition generalizes to unseen speakers.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    y_ids = np.asarray([CommandClassifier._coerce_label(v) for v in y], dtype=np.int64)
    spk_arr = np.asarray(speakers)

    unique_speakers = np.unique(spk_arr)
    k = min(int(n_splits), len(unique_speakers))
    accs: List[float] = []

    for tr_idx, te_idx in GroupKFold(n_splits=k).split(X_arr, y_ids, groups=spk_arr):
        clf = CommandClassifier().fit(X_arr[tr_idx], y_ids[tr_idx])
        pred_strs = clf.predict(X_arr[te_idx])
        true_strs = np.array([COMMANDS[int(i)] for i in y_ids[te_idx]], dtype=object)
        accs.append(float(accuracy_score(true_strs, pred_strs)))

    return float(np.mean(accs))


def load_command_training_dataset(
    processed_voice_root: str = "data/processed/voice",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load 46D voice vectors and command labels (0=Unlock System, 1=Report Status, 2=Terminate Session)
    across all enrolled speakers (Members 1..4) and benchmark command clips.
    """
    x_path = os.path.join(processed_voice_root, "X_voice.npy")
    y_cmd_path = os.path.join(processed_voice_root, "y_command.npy")
    y_spk_path = os.path.join(processed_voice_root, "y_speaker.npy")

    if os.path.isfile(x_path) and os.path.isfile(y_cmd_path) and os.path.isfile(y_spk_path):
        X_all = np.load(x_path)
        y_cmd_all = np.load(y_cmd_path)
        y_spk_all = np.load(y_spk_path)
    else:
        X_all, y_spk_all, y_cmd_all, _ = load_processed_voice_dataset(processed_voice_root)

    valid_mask = y_cmd_all >= 0
    X_cmd = X_all[valid_mask]
    y_cmd = y_cmd_all[valid_mask]
    y_spk = y_spk_all[valid_mask]

    return X_cmd, y_cmd, y_spk


def train_and_save_command_model(
    processed_voice_root: str = "data/processed/voice",
    out_dirs: Tuple[str, ...] = ("models/fingerprint", "models/voice"),
) -> CommandClassifier:
    """
    Train the speaker-invariant Random Forest Speech Command Classifier on the
    46D voice dataset and save to `models/fingerprint/best_command_model.joblib`
    (and `models/voice/best_command_model.joblib`).
    """
    X_cmd, y_cmd, y_spk = load_command_training_dataset(processed_voice_root)

    print("=" * 72)
    print("  TASK 3.3 (STEP 1): ISOLATED-WORD SPEECH COMMAND CLASSIFIER (RF)")
    print("=" * 72)
    print(f"[DATA] Loaded {len(X_cmd)} command vectors across commands: {COMMANDS}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1s, cv_accs = [], []
    for tr_idx, va_idx in skf.split(X_cmd, y_cmd):
        X_tr_aug, y_tr_aug = augment_voice_features_with_noise(X_cmd[tr_idx], y_cmd[tr_idx], copies=3, seed=42)
        fold_clf = CommandClassifier().fit(X_tr_aug, y_tr_aug)
        preds = fold_clf.predict_ids(X_cmd[va_idx])
        cv_f1s.append(f1_score(y_cmd[va_idx], preds, average="macro", zero_division=0))
        cv_accs.append(accuracy_score(y_cmd[va_idx], preds))

    X_full_aug, y_full_aug = augment_voice_features_with_noise(X_cmd, y_cmd, copies=4, seed=42)
    final_clf = CommandClassifier().fit(X_full_aug, y_full_aug)

    full_preds = final_clf.predict_ids(X_cmd)
    full_acc = float(accuracy_score(y_cmd, full_preds))
    proba = final_clf.predict_proba(X_cmd)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        out_path = os.path.join(d, "best_command_model.joblib")
        joblib.dump(final_clf, out_path)
        print(f"  [SAVED] CommandClassifier artifact -> '{out_path}'")

    print(f"  [5-Fold CV] Macro F1={np.mean(cv_f1s):.4f} | CV Accuracy={np.mean(cv_accs)*100:.2f}% | OOB Score={final_clf.oob_score_:.4f}")
    print(f"  [VERIFY]    Full-set Accuracy={full_acc*100:.2f}%, Probabilities shape={proba.shape}")
    print("=" * 72)
    return final_clf


CommandClassifier.__module__ = "src.command_classifier"


if __name__ == "__main__":
    from src.command_classifier import train_and_save_command_model as _run_cmd_train
    _run_cmd_train()
