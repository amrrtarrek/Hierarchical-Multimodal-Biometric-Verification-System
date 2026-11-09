"""
tests/test_voice_classifiers.py
Unit tests for Task 2.3: Speaker Verification Models (SVM & Logistic Regression).
"""
import os
import joblib
import numpy as np
import pytest

from src.base_interfaces import BaseBiometricModel
from src.voice_classifiers import (
    SpeakerLogisticRegressionModel,
    SpeakerSVMModel,
    augment_voice_features_with_noise,
    augment_waveform_with_ambient_noise,
    compute_far_frr,
    train_and_select_best_voice_model,
)


def make_synthetic_voice_dataset(n_per_class: int = 12, n_classes: int = 5, n_features: int = 46):
    rng = np.random.default_rng(42)
    X_list, y_list = [], []
    for cls in range(n_classes):
        center = rng.normal(cls * 2.5, 0.5, size=(1, n_features)).astype(np.float32)
        samples = center + rng.normal(0.0, 0.35, size=(n_per_class, n_features)).astype(np.float32)
        X_list.append(samples)
        y_list.append(np.full(n_per_class, cls, dtype=np.int64))
    return np.vstack(X_list), np.concatenate(y_list)


class TestTask23SpeakerVerificationModels:
    def test_models_inherit_from_base_biometric_model(self):
        assert isinstance(SpeakerSVMModel(), BaseBiometricModel)
        assert isinstance(SpeakerLogisticRegressionModel(), BaseBiometricModel)

    @pytest.mark.parametrize("kernel,degree", [("linear", 1), ("poly", 2), ("poly", 3)])
    def test_speaker_svm_kernels_and_platt_calibration(self, kernel, degree):
        X, y = make_synthetic_voice_dataset(n_per_class=10, n_classes=5)
        clf = SpeakerSVMModel(kernel=kernel, C=1.0, tol=1e-3, degree=degree).fit(X, y)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)

        assert preds.shape == (len(X),)
        assert proba.shape == (len(X), 5)
        assert proba.dtype == np.float32
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert 0 in clf.classes_

    def test_speaker_logistic_regression_ovr_l2_baseline(self):
        X, y = make_synthetic_voice_dataset(n_per_class=10, n_classes=5)
        clf = SpeakerLogisticRegressionModel(C=1.0, tol=1e-4).fit(X, y)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)

        assert preds.shape == (len(X),)
        assert proba.shape == (len(X), 5)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_dual_interface_artifact_compatibility(self):
        """Verify model supports both direct .predict_proba(X) AND dict-key lookup artifact['model']."""
        X, y = make_synthetic_voice_dataset(n_per_class=8, n_classes=3)
        clf = SpeakerSVMModel(kernel="linear", C=1.0).fit(X, y)

        # Direct BaseBiometricModel interface (Member 4)
        p_direct = clf.predict_proba(X[:2])
        assert p_direct.shape == (2, 3)

        # Dictionary-style lookup interface (Member 1 / Member 3)
        assert "model" in clf
        assert clf["model"] is clf
        assert "label_names" in clf
        assert clf["classes"] == [0, 1, 2]

    def test_ambient_noise_augmentation_shapes(self):
        X, y = make_synthetic_voice_dataset(n_per_class=6, n_classes=3)
        X_aug, y_aug = augment_voice_features_with_noise(X, y, copies=3)
        assert X_aug.shape == (len(X) * 4, 46)
        assert y_aug.shape == (len(y) * 4,)
        assert X_aug.dtype == np.float32

        raw_wave = np.sin(np.linspace(0, 50 * np.pi, 24000)).astype(np.float32) * 0.5
        noisy_wave = augment_waveform_with_ambient_noise(raw_wave, snr_db=15.0)
        assert noisy_wave.shape == (24000,)
        assert noisy_wave.dtype == np.float32

    def test_serialized_best_voice_model_definition_of_done(self):
        """Verify models/voice/best_voice_model.joblib exists and distinguishes enrolled vs Class 0."""
        model_path = "models/voice/best_voice_model.joblib"
        if not os.path.isfile(model_path):
            train_and_select_best_voice_model()

        assert os.path.isfile(model_path)
        loaded = joblib.load(model_path)

        X_voice = np.load("data/processed/voice/X_voice.npy")
        y_speaker = np.load("data/processed/voice/y_speaker.npy")

        proba = loaded.predict_proba(X_voice)
        preds = loaded.predict(X_voice)

        assert proba.shape == (len(X_voice), 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        far, frr = compute_far_frr(y_speaker, preds)
        assert far <= 0.10, f"Expected low Class 0 FAR, got {far:.4f}"
        assert frr <= 0.10, f"Expected low genuine FRR, got {frr:.4f}"

