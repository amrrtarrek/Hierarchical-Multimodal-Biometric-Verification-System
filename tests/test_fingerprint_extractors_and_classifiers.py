"""
tests/test_fingerprint_extractors_and_classifiers.py
Unit tests for Task 3.2: Fingerprint 2D Gabor Filter Bank Extraction & Model.
"""
import os
import joblib
import numpy as np
import pytest

from src.base_interfaces import BaseBiometricModel, BaseFeatureExtractor
from src.fingerprint_classifiers import (
    FingerprintSVMModel,
    compute_print_far_frr,
    train_and_select_best_fingerprint_model,
)
from src.fingerprint_extractors import (
    FINGERPRINT_FEATURE_DIM,
    N_GABORS,
    ORIENTATIONS,
    WAVELENGTHS,
    GaborFingerprintExtractor,
    build_gabor_filter_bank,
    build_gabor_kernel_2d,
)
from src.fingerprint_preprocessing import load_fingerprint_dataset


class TestTask32GaborFilterBankAndSVM:
    def test_inherits_from_base_interfaces(self):
        assert isinstance(GaborFingerprintExtractor(), BaseFeatureExtractor)
        assert isinstance(FingerprintSVMModel(), BaseBiometricModel)

    def test_gabor_bank_parameters_match_spec(self):
        """Step 1: 4 wavelengths {4, 8, 16, 24} x 8 orientations {0..7pi/8} = 32 kernels."""
        assert WAVELENGTHS == (4.0, 8.0, 16.0, 24.0)
        assert len(ORIENTATIONS) == 8
        assert ORIENTATIONS[0] == pytest.approx(0.0)
        assert ORIENTATIONS[1] == pytest.approx(np.pi / 8.0)
        assert ORIENTATIONS[-1] == pytest.approx(7.0 * np.pi / 8.0)
        assert N_GABORS == 32
        assert FINGERPRINT_FEATURE_DIM == 64

        bank = build_gabor_filter_bank()
        assert len(bank) == 32
        for k in bank:
            assert k.ndim == 2
            assert k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
            assert k.dtype == np.float32

    def test_gabor_extractor_output_shape_dtype_and_finite(self):
        """Steps 2 & 3: Output (N, 64) float32 array [mu_1, sigma_1, ..., mu_32, sigma_32]."""
        rng = np.random.default_rng(7)
        imgs = [rng.integers(0, 256, size=(128, 128), dtype=np.uint8) for _ in range(6)]
        ex = GaborFingerprintExtractor().fit(imgs)
        X = ex.transform(imgs)

        assert X.shape == (6, 64)
        assert X.dtype == np.float32
        assert np.all(np.isfinite(X))
        # Every odd index (1, 3, ..., 63) is a standard deviation sigma_k >= 0
        stds = X[:, 1::2]
        assert np.all(stds >= 0.0)

    def test_fingerprint_svm_probabilities_and_dual_interface(self):
        """Step 4: RBF-SVM with Platt calibration and dual dict/object interface."""
        rng = np.random.default_rng(11)
        X = rng.normal(size=(50, 64)).astype(np.float32)
        y = np.repeat([0, 1, 2, 3, 4], 10)
        for cls in range(5):
            X[y == cls] += cls * 2.5

        clf = FingerprintSVMModel(C=10.0, gamma="scale").fit(X, y)
        preds = clf.predict(X)
        proba = clf.predict_proba(X)

        assert preds.shape == (50,)
        assert proba.shape == (50, 5)
        assert proba.dtype == np.float32
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        # Dual interface check
        assert "model" in clf
        assert clf["model"] is clf
        assert "extractor" in clf

    def test_trained_best_fingerprint_model_definition_of_done(self):
        """Step 5 & Definition of Done: Verify serialized artifact distinguishes genuine vs Class 0."""
        model_path = "models/fingerprint/best_fingerprint_model.joblib"
        if not os.path.isfile(model_path):
            train_and_select_best_fingerprint_model()

        assert os.path.isfile(model_path)
        loaded = joblib.load(model_path)

        X_print = np.load("data/processed/fingerprint/X_print.npy")
        y_print = np.load("data/processed/fingerprint/y_print.npy")

        assert X_print.shape == (40, 64)
        assert X_print.dtype == np.float32
        assert np.all(np.isfinite(X_print))

        proba = loaded.predict_proba(X_print)
        preds = loaded.predict(X_print)
        assert proba.shape == (40, 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        far, frr = compute_print_far_frr(y_print, preds)
        assert far <= 0.05, f"Expected low Class 0 FAR, got {far:.4f}"
        assert frr <= 0.05, f"Expected low genuine FRR, got {frr:.4f}"

