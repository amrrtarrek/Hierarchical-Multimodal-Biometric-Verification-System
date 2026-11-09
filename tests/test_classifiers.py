"""
tests/test_classifiers.py
Unit tests verifying classifier shapes and probability calibration.
"""
import numpy as np
import pytest
from src.face_classifiers import FaceSVMModel, FaceRFModel, augment_dataset


def make_dataset(n_per_class=20, n_classes=5, n_features=576):
    X = np.random.randn(n_per_class * n_classes, n_features).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_per_class)
    return X, y


class TestFaceSVMModel:
    def test_fit_predict_shape(self):
        X, y = make_dataset()
        m = FaceSVMModel().fit(X, y)
        assert m.predict(X).shape == (len(X),)

    def test_predict_proba_shape(self):
        X, y = make_dataset(n_classes=3)
        m = FaceSVMModel().fit(X, y)
        assert m.predict_proba(X).shape == (len(X), 3)

    def test_proba_sums_to_one(self):
        X, y = make_dataset()
        m = FaceSVMModel().fit(X, y)
        np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1.0, atol=1e-5)

    def test_proba_all_positive(self):
        X, y = make_dataset()
        p = FaceSVMModel().fit(X, y).predict_proba(X)
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_impostor_class_zero(self):
        X, y = make_dataset(n_classes=4)
        m = FaceSVMModel().fit(X, y)
        assert 0 in m.classes_


class TestAugmentation:
    def test_augment_dataset_size_and_format(self):
        imgs = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(6)]
        labels = np.array([0, 0, 1, 1, 2, 2])
        a_imgs, a_labels = augment_dataset(imgs, labels, copies=3)
        assert len(a_imgs) == 6 * 4 and len(a_labels) == 6 * 4
        assert all(i.shape == (100, 100) and i.dtype == np.uint8 for i in a_imgs)
        assert set(a_labels) == {0, 1, 2}


class TestFaceRFModel:
    def test_oob_score_computed(self):
        X, y = make_dataset()
        m = FaceRFModel().fit(X, y)
        assert m.oob_score_ is not None and 0.0 <= m.oob_score_ <= 1.0

    def test_proba_sums_to_one(self):
        X, y = make_dataset()
        m = FaceRFModel().fit(X, y)
        np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1.0, atol=1e-5)