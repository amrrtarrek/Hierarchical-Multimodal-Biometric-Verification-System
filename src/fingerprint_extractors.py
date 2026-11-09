"""
src/fingerprint_extractors.py
Task 3.2 (Steps 1-3): Fingerprint 2D Gabor Filter Bank Extraction (Day 2 & 3 | Bonus 4).

Inherits from BaseFeatureExtractor (src/base_interfaces.py) enforcing:
- fit(samples) -> GaborFingerprintExtractor
- transform(samples) -> np.ndarray of shape (N, 64) and dtype float32

Mathematical Specifications (PDF Page 7-8):
1. 2D Gabor Filter Bank:
   - 4 spatial wavelengths: lambda in {4, 8, 16, 24} pixels
   - 8 orientations: theta in {0, pi/8, pi/4, 3pi/8, pi/2, 5pi/8, 3pi/4, 7pi/8}
   - Total: 4 x 8 = 32 spatial 2D Gabor kernels:
       G(x, y; theta, lambda, psi, sigma, gamma) =
           exp(-(x'^2 + gamma^2 * y'^2) / (2 * sigma^2)) * cos(2 * pi * (x' / lambda) + psi)
       where:
           x' =  x * cos(theta) + y * sin(theta)
           y' = -x * sin(theta) + y * cos(theta)
2. Convolve preprocessed 128x128 fingerprint crops with all 32 Gabor kernels.
3. Sub-band Spatial Statistical Descriptors (32 means + 32 stds = 64 features):
       mu_k = (1 / M) * sum_{i,j} I_k(i, j)
       sigma_k = sqrt((1 / M) * sum_{i,j} (I_k(i, j) - mu_k)^2)
       x_print = [mu_1, sigma_1, mu_2, sigma_2, ..., mu_32, sigma_32] in R^64
"""
from __future__ import annotations

from typing import Iterable, List, Tuple, Union

import cv2
import numpy as np

from src.base_interfaces import BaseFeatureExtractor
from src.fingerprint_preprocessing import FINGERPRINT_SIZE, preprocess

WAVELENGTHS: Tuple[float, ...] = (4.0, 8.0, 16.0, 24.0)
ORIENTATIONS: Tuple[float, ...] = tuple(k * np.pi / 8.0 for k in range(8))
N_GABORS: int = len(WAVELENGTHS) * len(ORIENTATIONS)  # 32 kernels
FINGERPRINT_FEATURE_DIM: int = N_GABORS * 2           # 64 features


def build_gabor_kernel_2d(
    wavelength: float,
    theta: float,
    sigma: float | None = None,
    gamma: float = 0.5,
    psi: float = 0.0,
    max_ksize: int = 31,
) -> np.ndarray:
    """
    Construct a 2D spatial Gabor kernel using the exact Task 3.2 mathematical callout:
        x' =  x * cos(theta) + y * sin(theta)
        y' = -x * sin(theta) + y * cos(theta)
        G(x, y) = exp(-(x'^2 + gamma^2 * y'^2) / (2 * sigma^2)) * cos(2 * pi * x' / lambda + psi)
    """
    lam = float(wavelength)
    if sigma is None:
        sigma = 0.56 * lam  # ~1 octave spatial frequency bandwidth

    half_size = min(max(3, int(np.ceil(3.0 * sigma))), max_ksize // 2)
    coords = np.arange(-half_size, half_size + 1, dtype=np.float64)
    y, x = np.meshgrid(coords, coords, indexing="ij")

    cos_t = np.cos(float(theta))
    sin_t = np.sin(float(theta))

    x_prime = x * cos_t + y * sin_t
    y_prime = -x * sin_t + y * cos_t

    gaussian_env = np.exp(-((x_prime ** 2) + (float(gamma) ** 2) * (y_prime ** 2)) / (2.0 * (float(sigma) ** 2)))
    carrier = np.cos(2.0 * np.pi * (x_prime / lam) + float(psi))
    kernel = (gaussian_env * carrier).astype(np.float32)

    # Zero-center kernel so DC background brightness does not dominate sub-band response
    kernel -= np.mean(kernel)
    return kernel


def build_gabor_filter_bank(
    wavelengths: Tuple[float, ...] = WAVELENGTHS,
    orientations: Tuple[float, ...] = ORIENTATIONS,
    gamma: float = 0.5,
    psi: float = 0.0,
) -> List[np.ndarray]:
    """Build all 32 2D Gabor kernels (4 wavelengths x 8 orientations)."""
    kernels: List[np.ndarray] = []
    for lam in wavelengths:
        for theta in orientations:
            kernels.append(
                build_gabor_kernel_2d(
                    wavelength=lam,
                    theta=theta,
                    gamma=gamma,
                    psi=psi,
                )
            )
    return kernels


def extract_gabor_features_single(
    img: np.ndarray,
    kernels: List[np.ndarray],
    target_size: Tuple[int, int] = FINGERPRINT_SIZE,
) -> np.ndarray:
    """
    Convolve a single 128x128 fingerprint image with all 32 Gabor kernels
    and return the interleaved 64D vector [mu_1, sigma_1, ..., mu_32, sigma_32].
    """
    if isinstance(img, str):
        loaded = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            raise FileNotFoundError(f"Could not read fingerprint image: {img}")
        gray = preprocess(loaded, binarize=True)
    else:
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.shape != target_size:
            arr = cv2.resize(arr.astype(np.uint8), target_size, interpolation=cv2.INTER_AREA)
        gray = arr

    norm_img = gray.astype(np.float32) / 255.0
    feats = np.empty(len(kernels) * 2, dtype=np.float32)

    for k_idx, kernel in enumerate(kernels):
        response = cv2.filter2D(norm_img, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        mu_k = float(np.mean(response))
        sigma_k = float(np.sqrt(np.mean((response - mu_k) ** 2)))
        feats[2 * k_idx] = mu_k
        feats[2 * k_idx + 1] = sigma_k

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats


class GaborFingerprintExtractor(BaseFeatureExtractor):
    """
    Task 3.2: 2D Gabor Filter Bank Feature Extractor implementing BaseFeatureExtractor.
    4 wavelengths x 8 orientations = 32 kernels; per response mean & std -> (N, 64) float32.
    """

    def __init__(
        self,
        wavelengths: Tuple[float, ...] = WAVELENGTHS,
        orientations: Tuple[float, ...] = ORIENTATIONS,
        gamma: float = 0.5,
        psi: float = 0.0,
        size: int = 128,
    ) -> None:
        self.wavelengths = tuple(wavelengths)
        self.orientations = tuple(orientations)
        self.gamma = float(gamma)
        self.psi = float(psi)
        self.size = int(size)
        self.kernels = build_gabor_filter_bank(
            wavelengths=self.wavelengths,
            orientations=self.orientations,
            gamma=self.gamma,
            psi=self.psi,
        )
        self.is_fitted_: bool = False

    def fit(self, samples: Iterable | None = None) -> "GaborFingerprintExtractor":
        """Stateless analytical filter bank; rebuilds kernels and marks fitted."""
        self.kernels = build_gabor_filter_bank(
            wavelengths=self.wavelengths,
            orientations=self.orientations,
            gamma=self.gamma,
            psi=self.psi,
        )
        self.is_fitted_ = True
        return self

    def transform(self, samples: Union[List, np.ndarray]) -> np.ndarray:
        """
        Transform a list/batch of 128x128 fingerprint images into a (N, 64) float32 matrix.
        """
        if isinstance(samples, np.ndarray) and samples.ndim == 2:
            sample_list = [samples]
        elif isinstance(samples, (str, bytes)):
            sample_list = [samples]
        else:
            sample_list = list(samples)

        n_samples = len(sample_list)
        if n_samples == 0:
            return np.empty((0, FINGERPRINT_FEATURE_DIM), dtype=np.float32)

        X = np.empty((n_samples, FINGERPRINT_FEATURE_DIM), dtype=np.float32)
        target_sz = (self.size, self.size)
        for i, s in enumerate(sample_list):
            X[i] = extract_gabor_features_single(s, self.kernels, target_size=target_sz)

        if not np.isfinite(X).all():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return X

