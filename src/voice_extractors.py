"""
src/voice_extractors.py
Task 2.2: Acoustic Feature Extraction (MFCC, ZCR, RMS Energy) (Day 2 | Core).

Inherits from BaseFeatureExtractor (src/base_interfaces.py) enforcing:
- fit(samples) -> VoiceFeatureExtractor
- transform(samples) -> np.ndarray of shape (N, 46) and dtype float32

Mathematical Specifications:
1. Framing & Windowing: 512-sample frames (32 ms at 16 kHz) with 256-sample hop (16 ms),
   windowed with a Hamming window.
2. MFCCs (40D):
   - Hertz to Mel: m = 2595 * log10(1 + f / 700), f = 700 * (10^(m / 2595) - 1)
   - STFT power spectrum filtered through K = 26 triangular Mel filterbanks
   - Log filterbank energy -> DCT-II:
       c_n = sum_{k=1}^{K} (log S_k) * cos(pi * n * (k - 0.5) / K), for n = 1..20
   - Temporal mean (20D) and standard deviation (20D) across frames -> 40D.
3. Zero-Crossing Rate (ZCR, 3D):
   - ZCR = (1 / (2N)) * sum_{t=1}^{N-1} |sgn(x[t]) - sgn(x[t-1])|
   - Mean, variance, and maximum across frames -> 3D.
4. Root-Mean-Square Energy (RMS, 3D):
   - RMS = sqrt((1 / N) * sum_{t=0}^{N-1} x[t]^2)
   - Mean, variance, and maximum across frames -> 3D.
5. Concatenation (46D):
   x_voice = [mu_mfcc (20), sigma_mfcc (20), mu_zcr, var_zcr, max_zcr, mu_rms, var_rms, max_rms]
"""
from __future__ import annotations

import os
from typing import Iterable, List, Tuple, Union

import numpy as np

from src.base_interfaces import BaseFeatureExtractor
from src.voice_preprocessing import (
    SAMPLE_RATE,
    TARGET_SAMPLES,
    condition_voice_waveform,
    decode_audio_file,
    load_conditioned_wav,
)

FRAME_LENGTH: int = 512      # 32 ms at 16 kHz
HOP_LENGTH: int = 256        # 16 ms at 16 kHz
N_FFT: int = 512
N_MELS: int = 26             # 26 triangular Mel filterbanks
N_MFCC: int = 20             # Retain lowest 20 cepstral coefficients (n = 1..20)
VOICE_FEATURE_DIM: int = 46  # 40 (MFCC) + 3 (ZCR) + 3 (RMS)
EPSILON: float = 1e-10


def hz_to_mel(freq_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert frequency in Hertz to Mel scale: m = 2595 * log10(1 + f / 700)."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Mel scale value to frequency in Hertz: f = 700 * (10^(m / 2595) - 1)."""
    return 700.0 * (np.power(10.0, np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def frame_audio(
    audio: np.ndarray,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Segment a 1D audio signal into overlapping frames of shape (n_frames, frame_length).
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(x) < frame_length:
        x = np.pad(x, (0, frame_length - len(x)), mode="constant", constant_values=0.0)

    n_frames = 1 + (len(x) - frame_length) // hop_length
    shape = (n_frames, frame_length)
    strides = (x.strides[0] * hop_length, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()


def build_mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """
    Construct `n_mels` (26) triangular Mel filterbanks over `(n_fft // 2 + 1)` STFT bins.
    Returns:
        filterbank matrix of shape (n_mels, n_fft // 2 + 1) and dtype float64.
    """
    if f_max is None:
        f_max = float(sample_rate) / 2.0

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = mel_to_hz(mel_points)

    fft_freqs = np.linspace(0.0, float(sample_rate) / 2.0, n_fft // 2 + 1, dtype=np.float64)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)

    for k in range(n_mels):
        f_left = hz_points[k]
        f_center = hz_points[k + 1]
        f_right = hz_points[k + 2]

        # Rising slope
        left_mask = (fft_freqs >= f_left) & (fft_freqs <= f_center)
        if f_center > f_left:
            fbank[k, left_mask] = (fft_freqs[left_mask] - f_left) / (f_center - f_left)

        # Falling slope
        right_mask = (fft_freqs >= f_center) & (fft_freqs <= f_right)
        if f_right > f_center:
            fbank[k, right_mask] = (f_right - fft_freqs[right_mask]) / (f_right - f_center)

    return fbank


def build_dct_matrix(n_mfcc: int = N_MFCC, n_mels: int = N_MELS) -> np.ndarray:
    """
    Construct the exact DCT-II matrix from the Task 2.2 formula callout:
        c_n = sum_{k=1}^{K} (log S_k) * cos(pi * n * (k - 0.5) / K), for n = 1, 2, ..., 20
    Returns:
        DCT matrix of shape (n_mels, n_mfcc) so `log_S @ dct_mat` yields (n_frames, n_mfcc).
    """
    k = np.arange(1, n_mels + 1, dtype=np.float64)[:, None]  # shape (K, 1), k = 1..26
    n = np.arange(1, n_mfcc + 1, dtype=np.float64)[None, :]  # shape (1, 20), n = 1..20
    dct_mat = np.cos(np.pi * n * (k - 0.5) / float(n_mels))
    return dct_mat


def compute_mfcc_features(
    frames: np.ndarray,
    hamming_window: np.ndarray,
    mel_fbank: np.ndarray,
    dct_mat: np.ndarray,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """
    Step 2: Compute frame-level 20D MFCCs and return concatenated temporal mean (20D)
    and temporal standard deviation (20D) -> 40D vector.
    """
    windowed = frames.astype(np.float64) * hamming_window[None, :]
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    power_spectrum = (np.abs(spectrum) ** 2) / float(n_fft)

    # Filter through 26 triangular Mel filterbanks: S_k (shape: n_frames, 26)
    mel_energies = np.matmul(power_spectrum, mel_fbank.T)
    log_mel_energies = np.log(np.maximum(mel_energies, EPSILON))

    # Apply DCT-II for n = 1..20: shape (n_frames, 20)
    mfcc_frames = np.matmul(log_mel_energies, dct_mat)

    mu_mfcc = np.mean(mfcc_frames, axis=0)
    sigma_mfcc = np.std(mfcc_frames, axis=0)
    return np.concatenate([mu_mfcc, sigma_mfcc]).astype(np.float32)


def compute_zcr_features(frames: np.ndarray) -> np.ndarray:
    """
    Step 3: Compute frame-level Zero-Crossing Rate (ZCR):
        ZCR = (1 / (2N)) * sum_{t=1}^{N-1} |sgn(x[t]) - sgn(x[t-1])|
    where sgn(x) = 1 if x >= 0 else -1.
    Returns [mean, variance, max] across frames -> 3D float32 vector.
    """
    n_samples = frames.shape[1]
    sgn = np.where(frames >= 0.0, 1.0, -1.0)
    zcr_per_frame = np.sum(np.abs(sgn[:, 1:] - sgn[:, :-1]), axis=1) / (2.0 * float(n_samples))

    mu_zcr = float(np.mean(zcr_per_frame))
    var_zcr = float(np.var(zcr_per_frame))
    max_zcr = float(np.max(zcr_per_frame))
    return np.array([mu_zcr, var_zcr, max_zcr], dtype=np.float32)


def compute_rms_features(frames: np.ndarray) -> np.ndarray:
    """
    Step 4: Compute frame-level Root-Mean-Square (RMS) Energy:
        RMS = sqrt((1 / N) * sum_{t=0}^{N-1} x[t]^2)
    Returns [mean, variance, max] across frames -> 3D float32 vector.
    """
    rms_per_frame = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

    mu_rms = float(np.mean(rms_per_frame))
    var_rms = float(np.var(rms_per_frame))
    max_rms = float(np.max(rms_per_frame))
    return np.array([mu_rms, var_rms, max_rms], dtype=np.float32)


def extract_voice_feature_vector(
    audio: np.ndarray,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    mel_fbank: np.ndarray | None = None,
    dct_mat: np.ndarray | None = None,
    hamming_window: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract the 46D acoustic feature vector for a single 1D audio signal:
        x_voice = [mu_mfcc (20D), sigma_mfcc (20D), mu_zcr, var_zcr, max_zcr, mu_rms, var_rms, max_rms]
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        x = np.zeros(TARGET_SAMPLES, dtype=np.float32)

    frames = frame_audio(x, frame_length=frame_length, hop_length=hop_length)
    if hamming_window is None:
        hamming_window = np.hamming(frame_length).astype(np.float64)
    if mel_fbank is None:
        mel_fbank = build_mel_filterbank(SAMPLE_RATE, frame_length, N_MELS)
    if dct_mat is None:
        dct_mat = build_dct_matrix(N_MFCC, N_MELS)

    mfcc_40d = compute_mfcc_features(frames, hamming_window, mel_fbank, dct_mat, n_fft=frame_length)
    zcr_3d = compute_zcr_features(frames)
    rms_3d = compute_rms_features(frames)

    feature_vec = np.concatenate([mfcc_40d, zcr_3d, rms_3d]).astype(np.float32)
    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feature_vec


class VoiceFeatureExtractor(BaseFeatureExtractor):
    """
    Task 2.2 Acoustic Feature Extractor implementing BaseFeatureExtractor.
    Transforms raw or conditioned audio arrays (or file paths) into (N, 46) float32 matrices.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_length: int = FRAME_LENGTH,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        n_mfcc: int = N_MFCC,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        self._hamming_window = np.hamming(self.frame_length).astype(np.float64)
        self._mel_fbank = build_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.frame_length,
            n_mels=self.n_mels,
        )
        self._dct_mat = build_dct_matrix(n_mfcc=self.n_mfcc, n_mels=self.n_mels)
        self.is_fitted_: bool = False

    def fit(self, samples: Iterable) -> "VoiceFeatureExtractor":
        """Fit extractor parameters (stateless analytical transform; marks extractor fitted)."""
        self._hamming_window = np.hamming(self.frame_length).astype(np.float64)
        self._mel_fbank = build_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.frame_length,
            n_mels=self.n_mels,
        )
        self._dct_mat = build_dct_matrix(n_mfcc=self.n_mfcc, n_mels=self.n_mels)
        self.is_fitted_ = True
        return self

    def _coerce_sample(self, sample: Union[str, np.ndarray, list]) -> np.ndarray:
        if isinstance(sample, str):
            if sample.lower().endswith(".wav"):
                try:
                    _, audio = load_conditioned_wav(sample)
                    if len(audio) == TARGET_SAMPLES:
                        return audio
                except Exception:
                    pass
            raw = decode_audio_file(sample, target_sr=self.sample_rate)
            return condition_voice_waveform(raw)
        arr = np.asarray(sample, dtype=np.float32).reshape(-1)
        return arr

    def transform(self, samples: Union[List, np.ndarray]) -> np.ndarray:
        """
        Transform a list/batch of audio arrays or file paths into a 2D float32 array of shape (N, 46).
        """
        if isinstance(samples, np.ndarray) and samples.ndim == 1:
            sample_list = [samples]
        elif isinstance(samples, (str, bytes)):
            sample_list = [samples]
        else:
            sample_list = list(samples)

        n_samples = len(sample_list)
        if n_samples == 0:
            return np.empty((0, VOICE_FEATURE_DIM), dtype=np.float32)

        out = np.empty((n_samples, VOICE_FEATURE_DIM), dtype=np.float32)
        for i, item in enumerate(sample_list):
            audio = self._coerce_sample(item)
            out[i] = extract_voice_feature_vector(
                audio,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                mel_fbank=self._mel_fbank,
                dct_mat=self._dct_mat,
                hamming_window=self._hamming_window,
            )
        return out


COMMAND_MAP = {
    "unlock": 0,     # "Unlock System"
    "status": 1,     # "Report Status"
    "terminate": 2,  # "Terminate Session"
}


def load_processed_voice_dataset(
    processed_voice_root: str = "data/processed/voice",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load all conditioned voice WAV files from `data/processed/voice`, extract 46D feature vectors,
    and return:
        X_voice: (N, 46) float32 feature matrix
        y_speaker: (N,) int64 speaker labels (0 = Class 0 Impostor, 1..4 = Members 1..4)
        y_command: (N,) int64 command labels (0 = Unlock System, 1 = Report Status, 2 = Terminate Session, -1 = N/A)
        file_paths: list of source WAV paths
    """
    extractor = VoiceFeatureExtractor().fit([])
    wav_paths: List[str] = []
    speaker_labels: List[int] = []
    command_labels: List[int] = []

    # Class 0: Impostors
    imp_dir = os.path.join(processed_voice_root, "class_0_impostor")
    if os.path.isdir(imp_dir):
        for fname in sorted(os.listdir(imp_dir)):
            if fname.lower().endswith(".wav"):
                wav_paths.append(os.path.join(imp_dir, fname))
                speaker_labels.append(0)
                command_labels.append(-1)

    # Classes 1..4: Enrolled Team Members
    for member_id in range(1, 5):
        m_dir = os.path.join(processed_voice_root, f"member_{member_id}")
        if not os.path.isdir(m_dir):
            continue
        for fname in sorted(os.listdir(m_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            wav_paths.append(os.path.join(m_dir, fname))
            speaker_labels.append(member_id)

            lower = fname.lower()
            cmd_id = -1
            for prefix, cid in COMMAND_MAP.items():
                if lower.startswith(prefix):
                    cmd_id = cid
                    break
            command_labels.append(cmd_id)

    X_voice = extractor.transform(wav_paths)
    y_speaker = np.asarray(speaker_labels, dtype=np.int64)
    y_command = np.asarray(command_labels, dtype=np.int64)
    return X_voice, y_speaker, y_command, wav_paths

