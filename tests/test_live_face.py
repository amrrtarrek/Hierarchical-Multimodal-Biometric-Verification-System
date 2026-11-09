"""
tests/test_live_face.py
Unit tests verifying live verification status thresholds and bounding boxes (task-sheet rules).
"""
import numpy as np
import pytest
from src.live_face import (draw_result, preprocess_crop, classify_status,
                           CONFIDENCE_VERIFIED, CONFIDENCE_AMBIGUOUS)


class TestPreprocessCrop:
    def test_output_shape(self):
        crop = np.random.randint(0, 256, (80, 60), dtype=np.uint8)
        assert preprocess_crop(crop).shape == (100, 100)

    def test_output_dtype(self):
        crop = np.random.randint(0, 256, (90, 70), dtype=np.uint8)
        assert preprocess_crop(crop).dtype == np.uint8


class TestDrawResult:
    """Pixels are sampled on the LEFT edge at mid-height (the label strip never covers it)."""

    def setup_method(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_green_box_on_verified(self):
        res = draw_result(self.frame.copy(), 100, 100, 150, 150, pred_class=1, name="Alice", confidence=0.90)
        px = res[175, 100]
        assert px[1] > 200 and px[2] < 50

    def test_red_box_on_impostor_class(self):
        res = draw_result(self.frame.copy(), 50, 100, 100, 100, pred_class=0, name="Impostor", confidence=0.95)
        px = res[150, 50]
        assert px[2] > 200 and px[1] < 50

    def test_red_box_on_low_confidence(self):
        res = draw_result(self.frame.copy(), 50, 100, 100, 100, pred_class=1, name="Alice", confidence=0.30)
        px = res[150, 50]
        assert px[2] > 200 and px[1] < 50

    def test_yellow_box_on_ambiguous(self):
        res = draw_result(self.frame.copy(), 200, 200, 100, 100, pred_class=1, name="Alice", confidence=0.70)
        px = res[250, 200]
        assert px[1] > 200 and px[2] > 200 and px[0] < 50


class TestStatusLogic:
    def test_thresholds_match_task_sheet(self):
        assert CONFIDENCE_VERIFIED == 0.85
        assert CONFIDENCE_AMBIGUOUS == 0.50

    @pytest.mark.parametrize("cls,p,expected", [
        (1, 0.85, "VERIFIED"), (1, 0.99, "VERIFIED"),
        (1, 0.84, "AMBIGUOUS"), (1, 0.50, "AMBIGUOUS"),
        (1, 0.49, "DENIED"), (1, 0.05, "DENIED"),
        (0, 0.99, "DENIED"), (0, 0.60, "DENIED"),
    ])
    def test_classify_status(self, cls, p, expected):
        assert classify_status(cls, p) == expected
