"""
src/voice_preprocessing.py
Task 2.1: Personal Voice Acquisition & Acoustic Conditioning (Day 1 | Bonus 3).

Implements:
1. Audio configuration & decoding: 16-bit PCM, single-channel (mono), 16,000 Hz sampling rate.
2. Voice Activity Detection (VAD) via short-term energy thresholding (onset/offset detection).
3. Strict temporal length enforcement: symmetric zero-padding or center-cropping to 1.5s (24,000 samples).
4. First-order pre-emphasis filter: y[t] = x[t] - 0.97 * x[t-1] to boost high-frequency formants.
5. Definition of Done verification: 1D float32 arrays of shape (24000,) with no clipping or DC offset.
"""
from __future__ import annotations

import os
import subprocess
import wave
from typing import Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE: int = 16000
TARGET_DURATION_SEC: float = 1.5
TARGET_SAMPLES: int = int(SAMPLE_RATE * TARGET_DURATION_SEC)  # 24,000 samples
PRE_EMPHASIS_ALPHA: float = 0.97

# Default VAD parameters (25 ms frame, 10 ms hop at 16 kHz)
VAD_FRAME_LENGTH: int = 400
VAD_HOP_LENGTH: int = 160
VAD_ENERGY_RATIO: float = 0.03
VAD_HANGOVER_FRAMES: int = 2


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset from a 1D audio signal by subtracting its mean."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    return (x - np.mean(x)).astype(np.float32)


def decode_audio_file(file_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Decode an audio/video container (.wav, .mpeg, .mp3, .mp4, .m4a, .ogg, .flac)
    into a 1D mono float32 waveform sampled at `target_sr` Hz in [-1.0, 1.0].
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    # Fast path for standard WAV files via scipy.io.wavfile
    if ext == ".wav":
        try:
            sr, data = wavfile.read(file_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if np.issubdtype(data.dtype, np.integer):
                info = np.iinfo(data.dtype)
                max_val = float(max(abs(info.min), abs(info.max)))
                audio = data.astype(np.float32) / max_val
            else:
                audio = data.astype(np.float32)
            if sr != target_sr:
                gcd = int(np.gcd(sr, target_sr))
                audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
            return remove_dc_offset(audio)
        except Exception:
            pass

    # Fallback / compressed media decoding (.mpeg, .mp4, .m4a, .mp3, etc.) via imageio-ffmpeg or system ffmpeg
    ffmpeg_exe = "ffmpeg"
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        file_path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-v",
        "quiet",
        "-",
    ]
    raw_bytes = subprocess.check_output(cmd)
    pcm = np.frombuffer(raw_bytes, dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    return remove_dc_offset(audio)


def detect_vad_bounds(
    audio: np.ndarray,
    frame_length: int = VAD_FRAME_LENGTH,
    hop_length: int = VAD_HOP_LENGTH,
    energy_threshold_ratio: float = VAD_ENERGY_RATIO,
    hangover_frames: int = VAD_HANGOVER_FRAMES,
) -> Tuple[int, int]:
    """
    Detect speech onset and offset sample indices using short-term energy thresholding.

    Returns:
        (onset_sample_idx, offset_sample_idx)
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    n_samples = len(x)
    if n_samples <= frame_length:
        return 0, n_samples

    # Compute short-term frame energies
    n_frames = 1 + (n_samples - frame_length) // hop_length
    energies = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = x[start : start + frame_length]
        energies[i] = np.mean(frame * frame)

    max_energy = float(np.max(energies))
    if max_energy <= 1e-10:
        return 0, n_samples

    # Estimate ambient room noise floor from the quietest 15% of frames
    sorted_energies = np.sort(energies)
    n_quiet = max(1, int(0.15 * n_frames))
    noise_floor = float(np.mean(sorted_energies[:n_quiet]))

    threshold = max(
        noise_floor * 3.0,
        noise_floor + energy_threshold_ratio * (max_energy - noise_floor),
        energy_threshold_ratio * max_energy,
    )

    active_frames = np.where(energies >= threshold)[0]
    if active_frames.size == 0:
        return 0, n_samples

    first_frame = max(0, int(active_frames[0]) - hangover_frames)
    last_frame = min(n_frames - 1, int(active_frames[-1]) + hangover_frames)

    onset_idx = first_frame * hop_length
    offset_idx = min(n_samples, last_frame * hop_length + frame_length)
    if offset_idx <= onset_idx:
        return 0, n_samples
    return onset_idx, offset_idx


def apply_vad(
    audio: np.ndarray,
    frame_length: int = VAD_FRAME_LENGTH,
    hop_length: int = VAD_HOP_LENGTH,
    energy_threshold_ratio: float = VAD_ENERGY_RATIO,
) -> np.ndarray:
    """
    Step 3: Strip leading and trailing room silence using short-term energy VAD.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    onset_idx, offset_idx = detect_vad_bounds(
        x,
        frame_length=frame_length,
        hop_length=hop_length,
        energy_threshold_ratio=energy_threshold_ratio,
    )
    return x[onset_idx:offset_idx].astype(np.float32)


def enforce_temporal_length(
    audio: np.ndarray,
    target_samples: int = TARGET_SAMPLES,
) -> np.ndarray:
    """
    Step 4: Symmetrically pad short clips with zeros or center-crop long clips
    to exactly `target_samples` (24,000 samples = 1.5s at 16 kHz).
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    current_len = len(x)

    if current_len == target_samples:
        return x.copy()
    elif current_len < target_samples:
        pad_total = target_samples - current_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(x, (pad_left, pad_right), mode="constant", constant_values=0.0).astype(np.float32)
    else:
        start = (current_len - target_samples) // 2
        return x[start : start + target_samples].astype(np.float32)


def apply_pre_emphasis(
    audio: np.ndarray,
    alpha: float = PRE_EMPHASIS_ALPHA,
) -> np.ndarray:
    """
    Step 5: Apply first-order pre-emphasis filter y[t] = x[t] - alpha * x[t-1]
    to boost high-frequency formants.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x.copy()
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y


def condition_voice_waveform(
    raw_audio: np.ndarray,
    target_samples: int = TARGET_SAMPLES,
    alpha: float = PRE_EMPHASIS_ALPHA,
    energy_threshold_ratio: float = VAD_ENERGY_RATIO,
    peak_ceiling: float = 0.92,
) -> np.ndarray:
    """
    Complete Task 2.1 acoustic conditioning pipeline on a raw 16 kHz mono array:
    1. Remove DC offset.
    2. Strip leading/trailing silence via Short-Term Energy VAD.
    3. Symmetrically pad or center-crop to 24,000 samples (1.5 seconds).
    4. Apply pre-emphasis filter y[t] = x[t] - 0.97 * x[t-1].
    5. Remove any post-filter DC offset and normalize peak amplitude safely below 1.0 (no clipping).
    """
    x = remove_dc_offset(raw_audio)
    trimmed = apply_vad(x, energy_threshold_ratio=energy_threshold_ratio)
    fixed_len = enforce_temporal_length(trimmed, target_samples=target_samples)
    emphasized = apply_pre_emphasis(fixed_len, alpha=alpha)
    centered = remove_dc_offset(emphasized)

    peak = float(np.max(np.abs(centered)))
    if peak > 1e-8:
        centered = (centered / peak) * peak_ceiling
        centered = remove_dc_offset(centered)

    return centered.astype(np.float32)


def save_wav_16bit_mono(file_path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """
    Save a conditioned 1D float32 waveform in [-1.0, 1.0] as a strict 16-bit PCM
    single-channel (mono) WAV file at 16,000 Hz, enforcing zero integer DC sum and no clipping.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    x = remove_dc_offset(audio)
    peak = float(np.max(np.abs(x)))
    if peak >= 0.999:
        x = (x / peak) * 0.95
        x = remove_dc_offset(x)

    pcm = np.clip(np.round(x * 32767.0), -32760, 32760).astype(np.int32)
    # Remove any integer rounding DC residual across active samples so raw int16 mean is ~0
    mean_int = int(np.round(np.mean(pcm)))
    if mean_int != 0:
        pcm = np.clip(pcm - mean_int, -32767, 32767)

    pcm_int16 = pcm.astype(np.int16)
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit (2 bytes)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())


def load_conditioned_wav(file_path: str) -> Tuple[int, np.ndarray]:
    """
    Load a conditioned 16-bit mono WAV file into a 1D float32 array in [-1.0, 1.0].
    """
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2 or n_channels != 1:
        raise ValueError(f"Expected 16-bit mono WAV, got {sampwidth * 8}-bit, {n_channels} channel(s): {file_path}")

    pcm = np.frombuffer(raw, dtype=np.int16)
    audio = (pcm.astype(np.float32) / 32768.0).astype(np.float32)
    return sr, audio


def condition_audio_file(
    input_path: str,
    output_wav_path: str,
    target_sr: int = SAMPLE_RATE,
    target_samples: int = TARGET_SAMPLES,
    alpha: float = PRE_EMPHASIS_ALPHA,
) -> np.ndarray:
    """
    Decode an input audio file, run full Task 2.1 conditioning, and write 16-bit mono WAV.
    """
    raw_audio = decode_audio_file(input_path, target_sr=target_sr)
    conditioned = condition_voice_waveform(
        raw_audio,
        target_samples=target_samples,
        alpha=alpha,
    )
    save_wav_16bit_mono(output_wav_path, conditioned, sample_rate=target_sr)
    return conditioned


def verify_voice_dataset(
    processed_voice_root: str = "data/processed/voice",
    target_samples: int = TARGET_SAMPLES,
    target_sr: int = SAMPLE_RATE,
    dc_tolerance: float = 1e-3,
) -> bool:
    """
    Definition of Done check for Task 2.1:
    Verify that all processed WAV files in `processed_voice_root` load as:
    - 16-bit PCM mono at 16,000 Hz
    - 1D float32 arrays of exact shape (24000,)
    - No clipping (max(|x|) < 1.0)
    - No DC offset (|mean(x)| <= dc_tolerance)
    - Non-silent and free of NaN/Inf
    """
    if not os.path.isdir(processed_voice_root):
        return False

    wav_files = []
    for root, _, files in os.walk(processed_voice_root):
        for f in sorted(files):
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    if not wav_files:
        return False

    for wav_path in wav_files:
        try:
            sr, audio = load_conditioned_wav(wav_path)
        except Exception:
            return False

        if sr != target_sr:
            return False
        if audio.ndim != 1 or audio.shape != (target_samples,):
            return False
        if audio.dtype != np.float32:
            return False
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            return False
        # No clipping check
        max_abs = float(np.max(np.abs(audio)))
        if max_abs >= 1.0 or max_abs <= 1e-5:
            return False
        # No DC offset check
        if abs(float(np.mean(audio))) > dc_tolerance:
            return False

    return True

