"""
Unit & Integration Tests for Member 4 Workstream (feature/fusion-eval):
- Task 4.1: Base Interfaces & Synthetic Fixtures
- Task 4.2: Early Feature Fusion (Per-Modality Z-Score & 686D SVM)
- Task 4.3: 3-Stage Hierarchical Verification Pipeline & RBAC
- Task 4.4: Evaluation Suite (FAR, FRR, EER, ROC/DET Curves & Confusion Matrices)
"""

from pathlib import Path

import numpy as np
import pytest

from early_fusion import EarlyFeatureFusionExtractor, EarlyFusionMultimodalModel
from evaluation import run_benchmark
from generate_fixtures import load_fixtures
from hierarchical_engine import HierarchicalVerificationEngine
from interfaces import (
    FACE_DIM,
    FUSED_DIM,
    IMPOSTOR_CLASS,
    PRINT_DIM,
    STATUS_PERMISSION_DENIED,
    STATUS_REJECTED_IMPOSTOR,
    STATUS_VERIFIED,
    VOICE_DIM,
    BaseBiometricModel,
    BaseFeatureExtractor,
)


class FixedProbModel(BaseBiometricModel):
    """Simple mock model returning fixed probability distributions for Stage 1-3 testing."""

    def __init__(self, probs, classes=(0, 1, 2, 3, 4)):
        self.probs = np.asarray(probs, dtype=np.float32).reshape(1, -1)
        self.classes_ = np.asarray(classes, dtype=np.int64)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([self.classes_[int(np.argmax(self.probs))]])

    def predict_proba(self, X):
        return self.probs


def test_task_4_1_interfaces_and_synthetic_fixtures():
    """Verify abstract interfaces cannot be instantiated directly and fixtures have exact shapes."""
    with pytest.raises(TypeError):
        BaseFeatureExtractor()
    with pytest.raises(TypeError):
        BaseBiometricModel()

    X_face, X_voice, X_print, y = load_fixtures("data/fixtures")
    n = len(y)
    assert X_face.shape == (n, FACE_DIM) and X_face.dtype == np.float32
    assert X_voice.shape == (n, VOICE_DIM) and X_voice.dtype == np.float32
    assert X_print.shape == (n, PRINT_DIM) and X_print.dtype == np.float32
    assert y.shape == (n,)
    assert IMPOSTOR_CLASS in y and 1 in y


def test_task_4_2_early_feature_fusion_zscore_and_svm():
    """Verify independent StandardScaler per modality -> 686D and calibrated SVM probabilities."""
    X_face, X_voice, X_print, y = load_fixtures("data/fixtures")

    extractor = EarlyFeatureFusionExtractor()
    X_fused = extractor.fit_transform((X_face, X_voice, X_print))

    assert X_fused.shape == (len(y), FUSED_DIM)
    assert X_fused.dtype == np.float32
    assert np.allclose(np.mean(X_fused, axis=0), 0.0, atol=1e-4)
    assert np.allclose(np.std(X_fused, axis=0), 1.0, atol=1e-3)

    model = EarlyFusionMultimodalModel(C=10.0, random_state=42).fit((X_face[:200], X_voice[:200], X_print[:200]), y[:200])
    probs = model.predict_proba((X_face[200:220], X_voice[200:220], X_print[200:220]))
    assert probs.shape[0] == 20
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_task_4_3_hierarchical_pipeline_stages():
    """Verify Stage 1 screening, Stage 2 voice + RBAC, and Stage 3 multimodal arbitration."""
    xf = np.ones(FACE_DIM, dtype=np.float32)
    xv = np.ones(VOICE_DIM, dtype=np.float32)
    xp = np.ones(PRINT_DIM, dtype=np.float32)
    dummy = FixedProbModel([0.2, 0.2, 0.2, 0.2, 0.2])

    # 1. Stage 1 Fast Verification (P_face = 0.92 >= 0.88)
    eng_s1_ok = HierarchicalVerificationEngine(FixedProbModel([0.02, 0.92, 0.02, 0.02, 0.02]), dummy, dummy)
    res1 = eng_s1_ok.verify(xf)
    assert res1.stage == 1 and res1.status == STATUS_VERIFIED and res1.identity == 1

    # 2. Stage 1 Fast Impostor Rejection (P_face = 0.15 < 0.35)
    eng_s1_rej = HierarchicalVerificationEngine(FixedProbModel([0.75, 0.15, 0.04, 0.03, 0.03]), dummy, dummy)
    res2 = eng_s1_rej.verify(xf)
    assert res2.stage == 1 and res2.status == STATUS_REJECTED_IMPOSTOR and res2.identity == 0

    # 3. Stage 2 Escalation (P_face = 0.65) + Voice Match (P_voice = 0.82 >= 0.75) + RBAC Check
    face_ambig_m4 = FixedProbModel([0.10, 0.05, 0.05, 0.15, 0.65])  # ID=4 (Auditor)
    voice_match_m4 = FixedProbModel([0.03, 0.03, 0.04, 0.08, 0.82])  # ID=4, P_voice=0.82
    eng_s2 = HierarchicalVerificationEngine(face_ambig_m4, voice_match_m4, dummy)

    res_s2_ok = eng_s2.verify(xf, xv, command="Report Status")
    assert res_s2_ok.stage == 2 and res_s2_ok.status == STATUS_VERIFIED and res_s2_ok.identity == 4

    res_s2_deny = eng_s2.verify(xf, xv, command="Terminate Session")
    assert res_s2_deny.stage == 2 and res_s2_deny.status == STATUS_PERMISSION_DENIED and res_s2_deny.identity == 4

    # 4. Stage 3 Escalation (Voice P_voice = 0.55 < 0.75) -> Multimodal Fusion Arbitration (P_fused = 0.78 >= 0.60)
    voice_low = FixedProbModel([0.15, 0.10, 0.10, 0.10, 0.55])
    fusion_ok = FixedProbModel([0.05, 0.05, 0.06, 0.06, 0.78])
    eng_s3 = HierarchicalVerificationEngine(face_ambig_m4, voice_low, fusion_ok)
    res_s3 = eng_s3.verify(xf, xv, xp)
    assert res_s3.stage == 3 and res_s3.status == STATUS_VERIFIED and res_s3.identity == 4


def test_task_4_4_evaluation_metrics_and_benchmark_outputs():
    """Verify FAR/FRR/EER math and ensure Multimodal Early Fusion achieves lower EER than unimodal baselines."""
    res_map = run_benchmark()

    assert res_map["Early Fusion"].eer < res_map["Face-Only"].eer
    assert res_map["Early Fusion"].eer < res_map["Voice-Only"].eer
    assert res_map["Early Fusion"].eer < res_map["Fingerprint-Only"].eer

    for fname in ("confusion_matrices.png", "roc_curves.png", "det_curves.png", "eer_comparison_table.md"):
        assert (Path("reports/figures") / fname).exists()
