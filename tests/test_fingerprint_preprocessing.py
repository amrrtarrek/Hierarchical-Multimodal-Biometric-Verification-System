"""
tests/test_fingerprint_preprocessing.py
Unit tests for Task 3.1: Personal Fingerprint Data Acquisition & Preprocessing.
"""
import os
import cv2
import numpy as np
import pytest

from src.fingerprint_preprocessing import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    FINGERPRINT_SIZE,
    crop_fingerprint_roi,
    load_fingerprint_dataset,
    preprocess,
    preprocess_fingerprint_image,
    verify_fingerprint_dataset,
)


class TestTask31FingerprintPreprocessing:
    def test_constants_match_task_specification(self):
        assert FINGERPRINT_SIZE == (128, 128)
        assert CLAHE_CLIP_LIMIT == pytest.approx(2.0)
        assert CLAHE_TILE_GRID_SIZE == (8, 8)

    def test_crop_fingerprint_roi_strips_blank_borders(self):
        canvas = np.full((200, 200), 245, dtype=np.uint8)
        # Active ridge pattern in center [40:160, 40:160]
        yy, xx = np.meshgrid(np.linspace(-1, 1, 120), np.linspace(-1, 1, 120), indexing="ij")
        ridges = (128 + 100 * np.cos(2 * np.pi * 10 * xx)).astype(np.uint8)
        canvas[40:160, 40:160] = ridges

        cropped = crop_fingerprint_roi(canvas)
        assert cropped.shape[0] < 180 and cropped.shape[1] < 180
        assert cropped.shape[0] >= 100 and cropped.shape[1] >= 100

    def test_preprocess_returns_128x128_binarized_and_clahe_modes(self):
        yy, xx = np.meshgrid(np.linspace(-1, 1, 160), np.linspace(-1, 1, 150), indexing="ij")
        raw = (128 + 90 * np.sin(2 * np.pi * 8 * (xx + 0.3 * yy))).astype(np.uint8)

        # Default (Otsu binarized ridge map)
        bin_img = preprocess_fingerprint_image(raw, binarize=True)
        assert bin_img is not None
        assert bin_img.shape == (128, 128)
        assert bin_img.dtype == np.uint8
        assert set(np.unique(bin_img)).issubset({0, 255})
        assert float(np.std(bin_img)) > 50.0

        # Continuous CLAHE grayscale mode
        clahe_img = preprocess_fingerprint_image(raw, binarize=False)
        assert clahe_img is not None
        assert clahe_img.shape == (128, 128)
        assert clahe_img.dtype == np.uint8

    def test_returns_none_on_blank_image(self):
        blank = np.zeros((150, 150), dtype=np.uint8)
        assert preprocess_fingerprint_image(blank) is None

    def test_processed_fingerprint_dataset_definition_of_done(self):
        processed_root = "data/processed/fingerprint"
        assert os.path.isdir(processed_root), "Run process_fingerprint_data.py first."
        assert verify_fingerprint_dataset(processed_root) is True

        images, labels, paths = load_fingerprint_dataset(processed_root, include_benchmark=False)
        assert len(images) == 40  # 20 Class 0 (10 subjects x 2) + 20 Enrolled (4 members x 5)
        assert set(np.unique(labels)) == {0, 1, 2, 3, 4}
        assert all(img.shape == (128, 128) and img.dtype == np.uint8 for img in images)

        # Verify Member 3 has 5 impressions
        m3_dir = os.path.join(processed_root, "member_3")
        m3_files = [f for f in os.listdir(m3_dir) if f.endswith(".png")]
        assert len(m3_files) == 5

