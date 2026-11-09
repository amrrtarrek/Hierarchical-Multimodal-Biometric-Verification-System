"""
tests/test_voice_preprocessing.py
Unit tests for Task 2.1: Personal Voice Acquisition & Acoustic Conditioning.
"""
import os
import numpy as np
import pytest

from src.voice_preprocessing import (
    PRE_EMPHASIS_ALPHA,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    apply_pre_emphasis,
    apply_vad,
    condition_audio_file,
    condition_voice_waveform,
    detect_vad_bounds,
    enforce_temporal_length,
    load_conditioned_wav,
    remove_dc_offset,
    save_wav_16bit_mono,
    verify_voice_dataset,
)


class TestTask21AcousticConditioning:
    def test_constants_match_task_specification(self):
        assert SAMPLE_RATE == 16000
        assert TARGET_SAMPLES == 24000
        assert PRE_EMPHASIS_ALPHA == pytest.approx(0.97)

    def test_remove_dc_offset(self):
        sig = np.ones(1000, dtype=np.float32) * 0.35 + np.sin(np.linspace(0, 20, 1000)).astype(np.float32) * 0.2
        centered = remove_dc_offset(sig)
        assert centered.dtype == np.float32
        assert abs(float(np.mean(centered))) < 1e-6

    def test_vad_strips_leading_and_trailing_silence(self):
        rng = np.random.default_rng(42)
        silence_lead = rng.normal(0.0, 0.001, 8000).astype(np.float32)
        t = np.arange(12000, dtype=np.float32) / SAMPLE_RATE
        speech = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        silence_trail = rng.normal(0.0, 0.001, 8000).astype(np.float32)
        full = np.concatenate([silence_lead, speech, silence_trail])

        onset, offset = detect_vad_bounds(full)
        assert 6000 <= onset <= 8500
        assert 19500 <= offset <= 22000

        trimmed = apply_vad(full)
        assert len(trimmed) < len(full)
        assert len(trimmed) >= 11000

    def test_enforce_temporal_length_symmetric_pad_short_clip(self):
        short_clip = np.ones(10000, dtype=np.float32) * 0.5
        padded = enforce_temporal_length(short_clip, target_samples=24000)
        assert padded.shape == (24000,)
        assert padded.dtype == np.float32
        # 14,000 zeros split symmetrically: 7,000 left and 7,000 right
        assert np.all(padded[:7000] == 0.0)
        assert np.all(padded[7000:17000] == 0.5)
        assert np.all(padded[17000:] == 0.0)

    def test_enforce_temporal_length_center_crop_long_clip(self):
        long_clip = np.arange(30000, dtype=np.float32)
        cropped = enforce_temporal_length(long_clip, target_samples=24000)
        assert cropped.shape == (24000,)
        assert cropped[0] == 3000.0
        assert cropped[-1] == 26999.0

    def test_pre_emphasis_formula_exact(self):
        x = np.array([1.0, 2.0, 0.5, -1.0], dtype=np.float32)
        y = apply_pre_emphasis(x, alpha=0.97)
        expected = np.array(
            [1.0, 2.0 - 0.97 * 1.0, 0.5 - 0.97 * 2.0, -1.0 - 0.97 * 0.5],
            dtype=np.float32,
        )
        np.testing.assert_allclose(y, expected, atol=1e-6)

    def test_full_waveform_conditioning_and_16bit_wav_io(self, tmp_path):
        t = np.arange(20000, dtype=np.float32) / SAMPLE_RATE
        raw = 0.25 + 0.6 * np.sin(2 * np.pi * 180.0 * t).astype(np.float32)
        conditioned = condition_voice_waveform(raw)
        assert conditioned.shape == (24000,)
        assert conditioned.dtype == np.float32
        assert float(np.max(np.abs(conditioned))) < 1.0
        assert abs(float(np.mean(conditioned))) < 1e-5

        wav_path = str(tmp_path / "test_out.wav")
        save_wav_16bit_mono(wav_path, conditioned)
        sr, loaded = load_conditioned_wav(wav_path)
        assert sr == 16000
        assert loaded.shape == (24000,)
        assert loaded.dtype == np.float32
        assert float(np.max(np.abs(loaded))) < 1.0
        assert abs(float(np.mean(loaded))) < 1e-4

    def test_processed_voice_dataset_definition_of_done(self):
        """Verify all conditioned WAVs in data/processed/voice meet Task 2.1 Definition of Done."""
        processed_root = "data/processed/voice"
        assert os.path.isdir(processed_root), "Run process_voice_data.py first."
        assert verify_voice_dataset(processed_root) is True

        # Check Member 2 specifically has 20 conditioned clips (7 unlock, 7 status, 6 terminate)
        m2_dir = os.path.join(processed_root, "member_2")
        m2_wavs = sorted([f for f in os.listdir(m2_dir) if f.endswith(".wav")])
        assert len(m2_wavs) == 20
        assert len([f for f in m2_wavs if f.startswith("unlock_")]) == 7
        assert len([f for f in m2_wavs if f.startswith("status_")]) == 7
        assert len([f for f in m2_wavs if f.startswith("terminate_")]) == 6

