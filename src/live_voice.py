"""
src/live_voice.py
Task 2.4: Real-Time Live Microphone Verification Stream (Day 4 & 5 | Bonus 2).

Procedural Steps:
1. Initialize non-blocking audio stream via `sounddevice` capturing sliding 100 ms
   chunks (1,600 samples) at 16 kHz.
2. Calculate short-time RMS energy; when energy exceeds ambient baseline, trigger
   capture mode and buffer 1.5 seconds (24,000 samples).
3. Apply pre-emphasis (y[t] = x[t] - 0.97 * x[t-1]), extract the 46D feature vector,
   and evaluate the trained speaker identity model (`models/voice/best_voice_model.joblib`).
4. Expose an active, thread-safe audio buffer queue (`ACTIVE_AUDIO_BUFFER_QUEUE`) for
   Member 3's command recognition engine (`rbac_engine.py`) to consume without race conditions.

Definition of Done:
Speaking into the microphone yields correct speaker identification within 500 ms of speech completion.
"""
from __future__ import annotations

import argparse
from collections import deque
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np

from src.voice_classifiers import (
    DEFAULT_SPEAKER_NAMES,
    SpeakerLogisticRegressionModel,
    SpeakerSVMModel,
)
from src.voice_extractors import VoiceFeatureExtractor, extract_voice_feature_vector
from src.voice_preprocessing import (
    PRE_EMPHASIS_ALPHA,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    apply_pre_emphasis,
    condition_voice_waveform,
    remove_dc_offset,
)

# Ensure pickled artifacts resolve cleanly even if trained under __main__
import __main__
for _cls in (SpeakerSVMModel, SpeakerLogisticRegressionModel, VoiceFeatureExtractor):
    setattr(__main__, _cls.__name__, _cls)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "voice", "best_voice_model.joblib")

CHUNK_DURATION_SEC: float = 0.100                      # 100 ms sliding chunks
CHUNK_SAMPLES: int = int(SAMPLE_RATE * CHUNK_DURATION_SEC)  # 1,600 samples at 16 kHz
CHUNKS_PER_UTTERANCE: int = TARGET_SAMPLES // CHUNK_SAMPLES  # 15 chunks = 1.5 seconds (24,000 samples)

CONFIDENCE_VERIFIED: float = 0.75    # Matches Stage 2 Voice verification threshold (P_voice >= 0.75)
CONFIDENCE_AMBIGUOUS: float = 0.50
MAX_IDENTIFICATION_LATENCY_MS: float = 500.0

DEFAULT_AMBIENT_RMS_FLOOR: float = 0.008
TRIGGER_MULTIPLIER: float = 2.5

# Module-level thread-safe queue exposed for Member 3's command recognition & RBAC engine
ACTIVE_AUDIO_BUFFER_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=64)


def compute_chunk_rms(chunk: np.ndarray) -> float:
    """
    Calculate short-time Root-Mean-Square (RMS) energy of an audio chunk:
        RMS = sqrt((1 / N) * sum_{t=0}^{N-1} x[t]^2)
    """
    x = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def classify_voice_status(pred_class: int, confidence: float) -> str:
    """
    Classify speaker verification decision status:
    - pred_class != 0 and confidence >= 0.75 -> 'VERIFIED'
    - pred_class != 0 and 0.50 <= confidence < 0.75 -> 'AMBIGUOUS'
    - pred_class == 0 or confidence < 0.50 -> 'DENIED'
    """
    if int(pred_class) == 0 or float(confidence) < CONFIDENCE_AMBIGUOUS:
        return "DENIED"
    if float(confidence) >= CONFIDENCE_VERIFIED:
        return "VERIFIED"
    return "AMBIGUOUS"


def format_terminal_log(record: Dict[str, Any]) -> str:
    """Format a single real-time speaker verification result for terminal logging."""
    status = record["status"]
    spk_id = record["speaker_id"]
    name = record["speaker_name"]
    conf_pct = record["confidence"] * 100.0
    lat_ms = record["latency_ms"]

    if status == "VERIFIED":
        tag = f"[VERIFIED]      Identity: {name} (ID={spk_id})"
    elif status == "AMBIGUOUS":
        tag = f"[AMBIGUOUS]     Escalating: {name} (ID={spk_id})"
    else:
        tag = f"[ACCESS DENIED] Impostor / Unverified (ID={spk_id})"

    return f"{tag:<48} | Score: {conf_pct:6.2f}% | Latency: {lat_ms:5.1f} ms"


def get_active_audio_queue() -> "queue.Queue[Dict[str, Any]]":
    """Return the shared thread-safe queue for Member 3's command recognition engine."""
    return ACTIVE_AUDIO_BUFFER_QUEUE


def consume_next_utterance(
    timeout: Optional[float] = 0.1,
    q: Optional["queue.Queue[Dict[str, Any]]"] = None,
) -> Optional[Dict[str, Any]]:
    """
    Thread-safe consumer helper for Member 3's speech command recognizer & RBAC engine.
    Returns the next buffered 1.5s utterance dictionary or None if the queue is empty.
    """
    target_q = q if q is not None else ACTIVE_AUDIO_BUFFER_QUEUE
    try:
        return target_q.get(block=timeout is not None and timeout > 0, timeout=timeout)
    except queue.Empty:
        return None


class LiveVoiceVerificationStream:
    """
    Real-Time Live Microphone Verification Stream (Task 2.4).

    Captures 100 ms chunks at 16 kHz via `sounddevice`, monitors short-time RMS energy
    against an adaptive ambient noise baseline, buffers 1.5s (24,000 samples) when speech
    onset is triggered, runs pre-emphasis + 46D feature extraction + speaker SVM inference,
    and pushes the utterance to a thread-safe queue for Member 3's RBAC engine.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
        ambient_rms_floor: float = DEFAULT_AMBIENT_RMS_FLOOR,
        trigger_multiplier: float = TRIGGER_MULTIPLIER,
        audio_queue: Optional["queue.Queue[Dict[str, Any]]"] = None,
        on_result_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.chunk_samples = int(chunk_samples)
        self.ambient_rms = float(ambient_rms_floor)
        self.trigger_multiplier = float(trigger_multiplier)
        self.audio_buffer_queue = audio_queue if audio_queue is not None else ACTIVE_AUDIO_BUFFER_QUEUE
        self.on_result_callback = on_result_callback

        self.extractor = VoiceFeatureExtractor(sample_rate=self.sample_rate).fit([])
        self.model = self._load_model(model_path)
        self.label_names: Dict[int, str] = getattr(
            self.model, "label_names", dict(DEFAULT_SPEAKER_NAMES)
        )

        # Internal capture state protected by a re-entrant lock
        self._lock = threading.RLock()
        self._pre_roll: deque = deque(maxlen=1)
        self._capturing: bool = False
        self._captured_chunks: list[np.ndarray] = []
        self._captured_samples: int = 0
        self._stream = None

    @staticmethod
    def _load_model(model_path: str):
        if os.path.isfile(model_path):
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict) and "model" in loaded:
                return loaded["model"]
            return loaded
        # Fallback untrained model only if artifact does not exist yet
        rng = np.random.default_rng(42)
        X_dummy = rng.standard_normal((25, 46)).astype(np.float32)
        y_dummy = np.repeat(np.arange(5), 5)
        return SpeakerSVMModel(kernel="linear", C=0.1).fit(X_dummy, y_dummy)

    @property
    def trigger_threshold(self) -> float:
        return max(self.ambient_rms * self.trigger_multiplier, DEFAULT_AMBIENT_RMS_FLOOR * 1.5)

    def calibrate_ambient_baseline(self, ambient_chunks: list[np.ndarray]) -> float:
        """Estimate ambient room noise RMS baseline from quiet calibration chunks."""
        if not ambient_chunks:
            return self.ambient_rms
        rms_vals = [compute_chunk_rms(c) for c in ambient_chunks]
        baseline = float(np.median(rms_vals))
        with self._lock:
            self.ambient_rms = max(baseline, 1e-4)
        return self.ambient_rms

    def evaluate_utterance(self, raw_utterance_24k: np.ndarray) -> Dict[str, Any]:
        """
        Step 3 & 4: Apply pre-emphasis & conditioning, extract 46D feature vector,
        evaluate trained speaker identity model, and push to thread-safe queue.
        """
        t0 = time.perf_counter()

        # Ensure exact 24,000 samples (1.5s), DC-offset removal, and pre-emphasis y[t] = x[t] - 0.97*x[t-1]
        x = remove_dc_offset(raw_utterance_24k)
        if len(x) < TARGET_SAMPLES:
            x = np.pad(x, (0, TARGET_SAMPLES - len(x)), mode="constant")
        elif len(x) > TARGET_SAMPLES:
            x = x[:TARGET_SAMPLES]

        emphasized = apply_pre_emphasis(x, alpha=PRE_EMPHASIS_ALPHA)
        conditioned = condition_voice_waveform(raw_utterance_24k, target_samples=TARGET_SAMPLES)

        # Extract 46D acoustic feature vector [mu_mfcc, sigma_mfcc, ZCR, RMS]
        features_46d = extract_voice_feature_vector(
            conditioned,
            mel_fbank=self.extractor._mel_fbank,
            dct_mat=self.extractor._dct_mat,
            hamming_window=self.extractor._hamming_window,
        )

        # Evaluate trained speaker verification model
        probs = np.asarray(self.model.predict_proba(features_46d.reshape(1, -1)), dtype=np.float32)[0]
        classes = np.asarray(self.model.classes_, dtype=np.int64)
        top_idx = int(np.argmax(probs))
        pred_speaker_id = int(classes[top_idx])
        confidence = float(probs[top_idx])

        status = classify_voice_status(pred_speaker_id, confidence)
        speaker_name = self.label_names.get(pred_speaker_id, f"Speaker {pred_speaker_id}")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        record: Dict[str, Any] = {
            "audio": conditioned,
            "pre_emphasized_audio": emphasized,
            "features_46d": features_46d,
            "speaker_id": pred_speaker_id,
            "speaker_name": speaker_name,
            "confidence": confidence,
            "probabilities": probs,
            "classes": classes,
            "status": status,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        }

        # Step 4: Enqueue into thread-safe buffer for Member 3's command recognizer without blocking
        try:
            self.audio_buffer_queue.put_nowait(record)
        except queue.Full:
            try:
                self.audio_buffer_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_buffer_queue.put_nowait(record)
            except queue.Full:
                pass

        if self.on_result_callback is not None:
            self.on_result_callback(record)

        return record

    def process_chunk(self, chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a single incoming 100 ms (1,600-sample) audio chunk.
        Triggers capture mode when chunk RMS > trigger_threshold and buffers 1.5 seconds.
        Returns the verification record when a 1.5s capture completes, else None.
        """
        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
        rms = compute_chunk_rms(mono)

        completed_utterance: Optional[np.ndarray] = None

        with self._lock:
            if not self._capturing:
                if rms > self.trigger_threshold:
                    self._capturing = True
                    self._captured_chunks = list(self._pre_roll) + [mono.copy()]
                    self._captured_samples = sum(len(c) for c in self._captured_chunks)
                else:
                    # Smoothly update ambient baseline during silence
                    self.ambient_rms = 0.95 * self.ambient_rms + 0.05 * max(rms, 1e-4)
                    self._pre_roll.append(mono.copy())
            else:
                self._captured_chunks.append(mono.copy())
                self._captured_samples += len(mono)

            if self._capturing and self._captured_samples >= TARGET_SAMPLES:
                full_buf = np.concatenate(self._captured_chunks, axis=0)[:TARGET_SAMPLES]
                completed_utterance = full_buf.astype(np.float32)
                self._capturing = False
                self._captured_chunks = []
                self._captured_samples = 0
                self._pre_roll.clear()

        if completed_utterance is not None:
            return self.evaluate_utterance(completed_utterance)
        return None

    def _sounddevice_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Non-blocking callback invoked by `sounddevice.InputStream` every 100 ms."""
        mono_chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        record = self.process_chunk(mono_chunk)
        if record is not None:
            print(format_terminal_log(record), flush=True)

    def start(self, duration_sec: Optional[float] = None, calibrate_sec: float = 0.5) -> None:
        """
        Start the non-blocking `sounddevice` microphone listener loop.
        Press Ctrl+C to stop (or pass `duration_sec` for a bounded session).
        """
        import sounddevice as sd

        print("=" * 78)
        print("  TASK 2.4: REAL-TIME LIVE MICROPHONE SPEAKER VERIFICATION STREAM")
        print("=" * 78)
        print(f"  Audio Config : {self.sample_rate} Hz, 16-bit/float32 Mono, {int(CHUNK_DURATION_SEC*1000)} ms chunks ({self.chunk_samples} samples)")
        print(f"  Model Loaded : {MODEL_PATH}")

        # Quick ambient noise calibration
        if calibrate_sec > 0:
            print(f"  [CALIBRATING] Measuring room ambient noise for {calibrate_sec:.1f}s (please stay quiet)...")
            n_cal_chunks = max(1, int(calibrate_sec / CHUNK_DURATION_SEC))
            cal_chunks = []
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=self.chunk_samples) as cal_stream:
                for _ in range(n_cal_chunks):
                    data, _ = cal_stream.read(self.chunk_samples)
                    cal_chunks.append(data.reshape(-1).copy())
            baseline = self.calibrate_ambient_baseline(cal_chunks)
            print(f"  [READY]       Ambient RMS={baseline:.5f} | Trigger Threshold={self.trigger_threshold:.5f}")

        print("  [LISTENING]   Speak a command ('Unlock System', 'Report Status', 'Terminate Session')... (Ctrl+C to exit)")
        print("-" * 78)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_samples,
            callback=self._sounddevice_callback,
        )
        start_time = time.time()
        with self._stream:
            try:
                while True:
                    time.sleep(0.05)
                    if duration_sec is not None and (time.time() - start_time) >= duration_sec:
                        break
            except KeyboardInterrupt:
                print("\n[STOPPED] Live microphone verification stream terminated by user.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2.4: Real-Time Live Microphone Speaker Verification")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds (default: infinite until Ctrl+C)")
    parser.add_argument("--simulate-file", type=str, default=None, help="Simulate real-time 100ms streaming from an audio file")
    args = parser.parse_args()

    verifier = LiveVoiceVerificationStream()

    if args.simulate_file:
        from src.voice_preprocessing import decode_audio_file
        raw = decode_audio_file(args.simulate_file, target_sr=SAMPLE_RATE)
        # Prepend 300ms silence + raw clip + pad to ensure 1.5s capture completes
        stream_sig = np.concatenate([
            np.zeros(CHUNK_SAMPLES * 3, dtype=np.float32),
            raw,
            np.zeros(TARGET_SAMPLES, dtype=np.float32),
        ])
        for start in range(0, len(stream_sig) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
            chunk = stream_sig[start : start + CHUNK_SAMPLES]
            rec = verifier.process_chunk(chunk)
            if rec is not None:
                print(format_terminal_log(rec))
                break
        return

    verifier.start(duration_sec=args.duration)


if __name__ == "__main__":
    main()

