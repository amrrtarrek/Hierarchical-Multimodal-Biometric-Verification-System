"""
tests/test_extractors.py
Unit tests verifying Task 1.2 Feature Extractor output shapes and dtypes.
"""
import numpy as np
import pytest
from src.face_extractors import HOGExtractor, LBPExtractor, PCAFaceExtractor


def make_dummy_images(n: int = 10) -> list:
    return [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(n)]


class TestHOGExtractor:
    def test_output_shape(self):
        imgs = make_dummy_images(10)
        ex = HOGExtractor().fit(imgs)
        out = ex.transform(imgs)
        assert out.shape == (10, 576)

    def test_output_dtype(self):
        imgs = make_dummy_images(5)
        out = HOGExtractor().fit(imgs).transform(imgs)
        assert out.dtype == np.float32

    def test_single_image(self):
        imgs = make_dummy_images(1)
        out = HOGExtractor().fit(imgs).transform(imgs)
        assert out.shape == (1, 576)

    def test_fit_transform_consistent(self):
        imgs = make_dummy_images(5)
        out1 = HOGExtractor().fit_transform(imgs)
        out2 = HOGExtractor().fit(imgs).transform(imgs)
        np.testing.assert_array_almost_equal(out1, out2)

    def test_no_nan_or_inf(self):
        imgs = make_dummy_images(5)
        out = HOGExtractor().fit(imgs).transform(imgs)
        assert not np.any(np.isnan(out)) and not np.any(np.isinf(out))


class TestLBPExtractor:
    def test_output_shape(self):
        imgs = make_dummy_images(10)
        out = LBPExtractor().fit(imgs).transform(imgs)
        assert out.shape == (10, 944)

    def test_output_dtype(self):
        imgs = make_dummy_images(5)
        out = LBPExtractor().fit(imgs).transform(imgs)
        assert out.dtype == np.float32

    def test_histograms_normalized(self):
        imgs = make_dummy_images(2)
        out = LBPExtractor().fit(imgs).transform(imgs)
        assert abs(out[0, :59].sum() - 1.0) < 0.01

    def test_no_nan(self):
        imgs = make_dummy_images(5)
        out = LBPExtractor().fit(imgs).transform(imgs)
        assert not np.any(np.isnan(out))


class TestPCAFaceExtractor:
    def test_output_shape(self):
        imgs = make_dummy_images(60)
        out = PCAFaceExtractor(n_components=50).fit(imgs).transform(imgs)
        assert out.shape == (60, 50)

    def test_output_dtype(self):
        imgs = make_dummy_images(60)
        out = PCAFaceExtractor(n_components=50).fit(imgs).transform(imgs)
        assert out.dtype == np.float32