"""
tests/test_preprocessing.py
Unit tests for Task 1.1 Preprocessing & Data Conditioning.
"""
import numpy as np
import cv2
import pytest
from src.face_preprocessing import preprocess_face_image, verify_dataset


@pytest.fixture
def real_face_fixture():
    for p in ["data/processed/subject_001/img_000.png", "data/processed/subject_002/img_000.png"]:
        img = cv2.imread(p)
        if img is not None:
            return img
    synthetic = np.zeros((200, 200, 3), dtype=np.uint8)
    synthetic[50:150, 50:150] = 180
    return synthetic


class TestPreprocessFaceImage:
    def test_returns_100x100_on_valid_face_image(self):
        img_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
        img_bgr[80:220, 80:220] = 180
        result = preprocess_face_image(img_bgr)
        if result is not None:
            assert result.shape == (100, 100)
            assert result.dtype == np.uint8
            assert 0 <= result.min() and result.max() <= 255

    def test_returns_none_on_blank_image(self):
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        assert preprocess_face_image(blank) is None

    def test_output_dtype_is_uint8(self, real_face_fixture):
        result = preprocess_face_image(real_face_fixture)
        if result is not None:
            assert result.dtype == np.uint8

    def test_pixel_range_valid(self, real_face_fixture):
        result = preprocess_face_image(real_face_fixture)
        if result is not None:
            assert result.min() >= 0 and result.max() <= 255


class TestVerifyDataset:
    def test_passes_on_correct_dataset(self, tmp_path):
        subj = tmp_path / "subject_001"
        subj.mkdir()
        for i in range(5):
            img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            cv2.imwrite(str(subj / f"img_{i:02d}.png"), img)
        assert verify_dataset(str(tmp_path)) is True

    def test_fails_on_wrong_shape(self, tmp_path):
        subj = tmp_path / "subject_001"
        subj.mkdir()
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(subj / "img_00.png"), img)
        assert verify_dataset(str(tmp_path)) is False

    def test_fails_on_empty_subject(self, tmp_path):
        (tmp_path / "subject_001").mkdir()
        assert verify_dataset(str(tmp_path)) is False