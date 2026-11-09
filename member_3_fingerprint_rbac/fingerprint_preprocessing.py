"""
src/fingerprint_preprocessing.py
Task 3.1: Personal Fingerprint Data Acquisition & Preprocessing (Day 1 | Bonus 3 & 4).

Procedural Steps:
1. Capture 5 high-resolution impressions per enrolled subject under varying contact
   pressure and angles (plus SOCOFing benchmark subjects & 10 external Class 0 subjects).
2. Crop to the active fingerprint pattern area (ROI) and resize uniformly to 128x128 pixels.
3. Apply Contrast Limited Adaptive Histogram Equalization (CLAHE with clipLimit=2.0,
   tileGridSize=(8, 8)) to enhance ridge-valley definition.
4. Apply Otsu's thresholding to generate binarized ridge maps while preserving core
   loop/delta singularities.
5. Verify Definition of Done: all processed prints load as clean 128x128 uint8 arrays
   with high ridge-valley contrast across all classes.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

SIZE: int = 128
FINGERPRINT_SIZE: Tuple[int, int] = (128, 128)
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)
MIN_CONTRAST_STD: float = 35.0


def crop_fingerprint_roi(gray: np.ndarray, grad_threshold_ratio: float = 0.12) -> np.ndarray:
    """
    Step 2: Isolate and crop to the active fingerprint ridge pattern area,
    removing blank scanner/camera borders.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return gray

    # Compute local gradient magnitude (Sobel) to locate active ridge-valley texture
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    smoothed_mag = cv2.GaussianBlur(mag, (15, 15), 0)

    max_mag = float(np.max(smoothed_mag))
    if max_mag <= 1e-3:
        return gray

    mask = smoothed_mag >= (grad_threshold_ratio * max_mag)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return gray

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    if (y_max - y_min) < int(0.25 * h) or (x_max - x_min) < int(0.25 * w):
        return gray

    return gray[y_min : y_max + 1, x_min : x_max + 1]


def preprocess_fingerprint_image(
    img: np.ndarray,
    binarize: bool = True,
    crop_roi: bool = True,
) -> Optional[np.ndarray]:
    """
    Complete Task 3.1 Fingerprint Conditioning Pipeline:
    1. Convert to 8-bit grayscale (`uint8`).
    2. Crop strictly to the fingerprint pattern area and resize uniformly to 128x128.
    3. Apply CLAHE (clipLimit=2.0, tileGridSize=(8, 8)) to enhance ridge-valley definition.
    4. Apply Otsu's thresholding (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`) to produce
       binarized ridge maps preserving core loop/delta singularities.
    Returns None if the input image is blank/uniform.
    """
    if img is None or img.size == 0:
        return None

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Reject completely blank or uniform images
    if float(np.std(gray)) < 1.0:
        return None

    # Step 2: Crop to fingerprint pattern area and resize uniformly to 128x128
    roi = crop_fingerprint_roi(gray) if crop_roi else gray
    resized = cv2.resize(roi, FINGERPRINT_SIZE, interpolation=cv2.INTER_AREA)

    # Step 3: Apply CLAHE (clipLimit=2.0, tileGridSize=(8, 8))
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    enhanced = clahe.apply(resized)

    # Step 4: Apply Otsu's thresholding to generate binarized ridge maps
    if binarize:
        _, binarized = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized.astype(np.uint8)

    return enhanced.astype(np.uint8)


def preprocess(img: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Convenience wrapper returning a 128x128 uint8 array (even on synthetic low-contrast test arrays).
    """
    res = preprocess_fingerprint_image(img, binarize=binarize, crop_roi=True)
    if res is not None:
        return res
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    resized = cv2.resize(gray, FINGERPRINT_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    enhanced = clahe.apply(resized)
    if binarize:
        _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return enhanced.astype(np.uint8)


def augment_fingerprint(img: np.ndarray, rng: np.random.Generator | np.random.RandomState) -> np.ndarray:
    """
    Simulate realistic fingerprint contact pressure and placement angle variations
    (rotation +/- 10 deg, shift +/- 5 px, morphological pressure dilation/erosion).
    """
    h, w = img.shape[:2]
    angle = float(rng.uniform(-10.0, 10.0))
    tx = float(rng.uniform(-5.0, 5.0))
    ty = float(rng.uniform(-5.0, 5.0))

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Simulate contact pressure variation before re-binarizing if grayscale
    gain = float(rng.uniform(0.88, 1.12))
    out = np.clip(warped.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if set(np.unique(img)).issubset({0, 255}):
        _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    return out


# Alias for backward compatibility
augment = augment_fingerprint


def load_fingerprint_dataset(
    processed_root: str = "data/processed/fingerprint",
    include_benchmark: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    Load processed 128x128 fingerprint images and integer class labels:
    - Class 0: `class_0_impostor` (10 external un-enrolled subjects)
    - Classes 1..4: `member_1`..`member_4` (enrolled team members)
    - Optional benchmark subjects (`benchmark/subject_*`)
    """
    images: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[str] = []

    if not os.path.isdir(processed_root):
        return images, np.array(labels, dtype=np.int64), paths

    # 1. Class 0: Impostors
    imp_dir = os.path.join(processed_root, "class_0_impostor")
    if os.path.isdir(imp_dir):
        for fname in sorted(os.listdir(imp_dir)):
            if fname.lower().endswith((".png", ".bmp", ".tif", ".jpg")):
                p = os.path.join(imp_dir, fname)
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(0)
                    paths.append(p)

    # 2. Classes 1..4: Enrolled Members
    for m_id in range(1, 5):
        m_dir = os.path.join(processed_root, f"member_{m_id}")
        if os.path.isdir(m_dir):
            for fname in sorted(os.listdir(m_dir)):
                if fname.lower().endswith((".png", ".bmp", ".tif", ".jpg")):
                    p = os.path.join(m_dir, fname)
                    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        labels.append(m_id)
                        paths.append(p)

    # 3. Optional SOCOFing Benchmark subjects
    if include_benchmark:
        bench_dir = os.path.join(processed_root, "benchmark")
        if os.path.isdir(bench_dir):
            for idx, fname in enumerate(sorted(os.listdir(bench_dir))):
                if fname.lower().endswith((".png", ".bmp", ".tif", ".jpg")):
                    p = os.path.join(bench_dir, fname)
                    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Parse subject ID from bench_subj_XX_...
                        parts = fname.split("_")
                        subj_num = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else (idx + 1)
                        images.append(img)
                        labels.append(4 + subj_num)
                        paths.append(p)

    return images, np.asarray(labels, dtype=np.int64), paths


def verify_fingerprint_dataset(processed_root: str = "data/processed/fingerprint") -> bool:
    """
    Definition of Done check for Task 3.1:
    All processed prints in `processed_root` load as clean 128x128 uint8 arrays
    with high ridge-valley contrast across all classes.
    """
    if not os.path.isdir(processed_root):
        return False

    found_files = 0
    for root, _, files in os.walk(processed_root):
        for f in sorted(files):
            if not f.lower().endswith(".png"):
                continue
            found_files += 1
            full_path = os.path.join(root, f)
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            if img.shape != FINGERPRINT_SIZE or img.dtype != np.uint8:
                return False
            # Verify high ridge-valley contrast (standard deviation >= MIN_CONTRAST_STD)
            if float(np.std(img)) < MIN_CONTRAST_STD:
                return False

    return found_files > 0

