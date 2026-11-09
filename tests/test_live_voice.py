"""
tests/test_live_voice.py
Unit tests for Task 2.4: Real-Time Live Microphone Verification Stream.
"""
import queue
import threading
import numpy as np
import pytest

from src.live_voice import (
    CHUNK_DURATION_SEC,
    CHUNK_SAMPLES,
    CONFIDENCE_AMBIGUOUS,
    CONFIDENCE_VERIFIED,
    MAX_IDENTIFICATION_LATENCY_MS,
    LiveVoiceVerificationStream,
    classify_voice_status,
    compute_chunk_rms,
    consume_next_utterance,
    format_terminal_log,
)
from src.voice_preprocessing import SAMPLE_RATE, TARGET_SAMPLES, load_conditioned_wav


class TestTask24LiveMicrophoneVerificationStream:
    def test_stream_constants_match_task_spec(self):
        """Step 1: 100 ms sliding chunks (1,600 samples) at 16 kHz, 1.5s buffer (24,000 samples)."""
        assert SAMPLE_RATE == 16000
        assert CHUNK_DURATION_SEC == pytest.approx(0.100)
        assert CHUNK_SAMPLES == 1600
        assert TARGET_SAMPLES == 24000
        assert CONFIDENCE_VERIFIED == pytest.approx(0.75)
        assert CONFIDENCE_AMBIGUOUS == pytest.approx(0.50)

    def test_compute_chunk_rms(self):
        silent = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        assert compute_chunk_rms(silent) == pytest.approx(0.0, abs=1e-7)

        const = np.full(CHUNK_SAMPLES, 0.4, dtype=np.float32)
        assert compute_chunk_rms(const) == pytest.approx(0.4, rel=1e-5)

    @pytest.mark.parametrize(
        "pred_cls,conf,expected",
        [
            (2, 0.92, "VERIFIED"),
            (1, 0.75, "VERIFIED"),
            (2, 0.65, "AMBIGUOUS"),
            (3, 0.50, "AMBIGUOUS"),
            (2, 0.42, "DENIED"),
            (0, 0.98, "DENIED"),
        ],
    )
    def test_classify_voice_status(self, pred_cls, conf, expected):
        assert classify_voice_status(pred_cls, conf) == expected

    def test_sliding_100ms_chunk_capture_and_latency_under_500ms(self):
        """
        Steps 2 & 3 + Definition of Done:
        Feed sliding 100ms chunks (silence -> Member 2 speech) and verify:
        - Silence chunks do not trigger capture
        - Speech onset triggers 1.5s capture (24,000 samples)
        - Member 2 is identified with latency < 500 ms
        """
        q = queue.Queue(maxsize=16)
        stream = LiveVoiceVerificationStream(audio_queue=q, ambient_rms_floor=0.005)

        # 4 chunks (400 ms) of quiet room silence -> no capture triggered
        rng = np.random.default_rng(10)
        for _ in range(4):
            quiet_chunk = rng.normal(0.0, 0.001, CHUNK_SAMPLES).astype(np.float32)
            assert stream.process_chunk(quiet_chunk) is None

        # Stream Member 2's clip (plus 1.5s trailing room buffer to complete 1.5s post-trigger window)
        _, m2_audio = load_conditioned_wav("data/processed/voice/member_2/status_01.wav")
        stream_signal = np.concatenate([m2_audio, np.zeros(TARGET_SAMPLES, dtype=np.float32)])
        result = None
        for i in range(0, len(stream_signal), CHUNK_SAMPLES):
            chunk = stream_signal[i : i + CHUNK_SAMPLES]
            rec = stream.process_chunk(chunk)
            if rec is not None:
                result = rec
                break

        assert result is not None
        assert result["audio"].shape == (24000,)
        assert result["features_46d"].shape == (46,)
        assert result["speaker_id"] == 2
        assert result["status"] == "VERIFIED"
        assert result["latency_ms"] < MAX_IDENTIFICATION_LATENCY_MS

        log_line = format_terminal_log(result)
        assert "VERIFIED" in log_line
        assert "Member 2" in log_line

    def test_thread_safe_audio_queue_for_member3_command_engine(self):
        """
        Step 4: Verify concurrent producer (live audio stream) and consumer
        (Member 3 command recognition engine) operate without race conditions.
        """
        shared_q = queue.Queue(maxsize=32)
        stream = LiveVoiceVerificationStream(audio_queue=shared_q)
        _, clip = load_conditioned_wav("data/processed/voice/member_2/status_01.wav")

        consumed_records = []

        def producer():
            for _ in range(5):
                stream.evaluate_utterance(clip)

        def consumer():
            while len(consumed_records) < 5:
                item = consume_next_utterance(timeout=0.5, q=shared_q)
                if item is not None:
                    consumed_records.append(item)

        t_prod = threading.Thread(target=producer)
        t_cons = threading.Thread(target=consumer)
        t_prod.start()
        t_cons.start()
        t_prod.join(timeout=5.0)
        t_cons.join(timeout=5.0)

        assert len(consumed_records) == 5
        assert shared_q.empty()
        for rec in consumed_records:
            assert rec["speaker_id"] == 2
            assert rec["features_46d"].shape == (46,)
            assert rec["latency_ms"] < 500.0
