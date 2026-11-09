"""
tests/test_voice_extractors.py
Unit tests for Task 2.2: Acoustic Feature Extraction (MFCC, ZCR, RMS Energy).
"""
import os
import numpy as np
import pytest

from src.base_interfaces import BaseFeatureExtractor
from src.voice_extractors import (
    FRAME_LENGTH,
    HOP_LENGTH,
    N_MFCC,
    N_MELS,
    VOICE_FEATURE_DIM,
    VoiceFeatureExtractor,
    build_dct_matrix,
    build_mel_filterbank,
    compute_mfcc_features,
    compute_rms_features,
    compute_zcr_features,
    extract_voice_feature_vector,
    frame_audio,
    hz_to_mel,
    load_processed_voice_dataset,
    mel_to_hz,
)
from src.voice_preprocessing import SAMPLE_RATE, TARGET_SAMPLES


def make_synthetic_clips(n: int = 8) -> list:
    rng = np.random.default_rng(123)
    t = np.arange(TARGET_SAMPLES, dtype=np.float32) / SAMPLE_RATE
    clips = []
    for i in range(n):
        freq = 120.0 + 35.0 * i
        sig = (
            0.5 * np.sin(2.0 * np.pi * freq * t)
            + 0.25 * np.sin(2.0 * np.pi * (2.0 * freq) * t)
            + 0.05 * rng.standard_normal(TARGET_SAMPLES)
        ).astype(np.float32)
        clips.append(sig)
    return clips


class TestTask22AcousticFeatureExtraction:
    def test_inherits_from_base_feature_extractor(self):
        ex = VoiceFeatureExtractor()
        assert isinstance(ex, BaseFeatureExtractor)

    def test_framing_parameters_match_spec(self):
        """Step 1: 512-sample frames (32 ms) with 256-sample hop (16 ms)."""
        assert FRAME_LENGTH == 512
        assert HOP_LENGTH == 256
        assert N_MELS == 26
        assert N_MFCC == 20
        assert VOICE_FEATURE_DIM == 46

        sig = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        frames = frame_audio(sig)
        expected_frames = 1 + (TARGET_SAMPLES - FRAME_LENGTH) // HOP_LENGTH  # 92 frames
        assert frames.shape == (expected_frames, 512)

    def test_hz_mel_conversion_roundtrip(self):
        """Verify exact Hertz <-> Mel formulas from Mathematical Callout."""
        freqs = np.array([0.0, 300.0, 700.0, 1000.0, 4000.0, 8000.0])
        mels = hz_to_mel(freqs)
        # At f = 700 Hz, log10(1 + 700/700) = log10(2), m = 2595 * log10(2)
        assert mels[2] == pytest.approx(2595.0 * np.log10(2.0), rel=1e-7)
        recovered = mel_to_hz(mels)
        np.testing.assert_allclose(recovered, freqs, atol=1e-7)

    def test_mel_filterbank_and_dct_matrix_shapes(self):
        """Step 2: 26 triangular Mel filterbanks and 20 DCT-II cepstral coefficients."""
        fbank = build_mel_filterbank(sample_rate=16000, n_fft=512, n_mels=26)
        assert fbank.shape == (26, 257)
        assert np.all(fbank >= 0.0) and np.all(fbank <= 1.0)
        # Every filter should have non-zero support
        assert np.all(np.sum(fbank, axis=1) > 0.0)

        dct_mat = build_dct_matrix(n_mfcc=20, n_mels=26)
        assert dct_mat.shape == (26, 20)
        # Check exact DCT-II entry for k=1, n=1: cos(pi * 1 * (1 - 0.5) / 26)
        expected_11 = np.cos(np.pi * 1.0 * 0.5 / 26.0)
        assert dct_mat[0, 0] == pytest.approx(expected_11, rel=1e-7)

    def test_zcr_and_rms_exact_formulas(self):
        """Steps 3 & 4: ZCR (3D) and RMS Energy (3D) exact formula verification."""
        # Alternating +1, -1 frame crosses zero at every sample step -> ZCR = (N - 1) / N
        alt_frame = np.array([[1.0 if i % 2 == 0 else -1.0 for i in range(512)]], dtype=np.float32)
        zcr_vec = compute_zcr_features(alt_frame)
        assert zcr_vec.shape == (3,)
        assert zcr_vec[0] == pytest.approx(511.0 / 512.0, rel=1e-5)
        assert zcr_vec[1] == pytest.approx(0.0, abs=1e-7)
        assert zcr_vec[2] == pytest.approx(511.0 / 512.0, rel=1e-5)

        # Constant amplitude 0.5 -> RMS = 0.5
        const_frame = np.full((4, 512), 0.5, dtype=np.float32)
        rms_vec = compute_rms_features(const_frame)
        assert rms_vec.shape == (3,)
        assert rms_vec[0] == pytest.approx(0.5, rel=1e-6)
        assert rms_vec[1] == pytest.approx(0.0, abs=1e-7)
        assert rms_vec[2] == pytest.approx(0.5, rel=1e-6)

    def test_output_shape_dtype_and_no_nan_or_inf(self):
        """Definition of Done: (N, 46) float32 matrix with zero NaN or Inf values."""
        clips = make_synthetic_clips(10)
        # Include an all-zero silent clip to verify log-energy epsilon prevents -Inf/NaN
        clips.append(np.zeros(TARGET_SAMPLES, dtype=np.float32))

        extractor = VoiceFeatureExtractor()
        X = extractor.fit_transform(clips)

        assert X.shape == (11, 46)
        assert X.dtype == np.float32
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_single_clip_and_fit_transform_consistency(self):
        clips = make_synthetic_clips(4)
        ex1 = VoiceFeatureExtractor().fit(clips)
        out1 = ex1.transform(clips)
        out2 = VoiceFeatureExtractor().fit_transform(clips)
        np.testing.assert_allclose(out1, out2, atol=1e-6)

        single = ex1.transform(clips[0])
        assert single.shape == (1, 46)
        assert single.dtype == np.float32

    def test_real_conditioned_voice_dataset_extraction(self):
        """Verify extraction on all conditioned WAVs in data/processed/voice."""
        if not os.path.isdir("data/processed/voice/member_2"):
            pytest.skip("Conditioned voice data not yet generated.")

        X_voice, y_speaker, y_command, paths = load_processed_voice_dataset("data/processed/voice")
        assert X_voice.ndim == 2
        assert X_voice.shape[1] == 46
        assert X_voice.shape[0] == len(y_speaker) == len(y_command) == len(paths)
        assert X_voice.dtype == np.float32
        assert not np.any(np.isnan(X_voice))
        assert not np.any(np.isinf(X_voice))
        # Ensure Class 0 (Impostor) and Enrolled Members 1..4 are present
        assert set(np.unique(y_speaker)) == {0, 1, 2, 3, 4}
        # Ensure all 3 RBAC commands (0, 1, 2) are present among enrolled clips
        assert {0, 1, 2}.issubset(set(np.unique(y_command)))

