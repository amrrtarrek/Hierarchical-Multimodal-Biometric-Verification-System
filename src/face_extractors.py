"""
src/face_extractors.py
Task 1.2 Mathematical Face Feature Extractors.
"""
import importlib
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import cv2
try:
    from skimage.feature import local_binary_pattern
except ImportError:
    local_binary_pattern = None
from sklearn.decomposition import PCA
from src.base_interfaces import BaseFeatureExtractor


def _numpy_nri_uniform_lbp_p8_r2(img: np.ndarray) -> np.ndarray:
    """Pure-NumPy fallback for P=8, R=2 non-rotation-invariant uniform LBP (59 bins: 0..58)."""
    # Precompute lookup table mapping 8-bit code (0..255) -> uniform bin (0..58)
    lut = np.full(256, 58, dtype=np.int32)
    u_idx = 0
    for code in range(256):
        bits = [(code >> b) & 1 for b in range(8)]
        transitions = sum(bits[b] != bits[(b + 1) % 8] for b in range(8))
        if transitions <= 2:
            lut[code] = u_idx
            u_idx += 1

    f_img = img.astype(np.float32)
    padded = np.pad(f_img, 2, mode="constant", constant_values=0.0)
    h, w = f_img.shape[:2]
    center = padded[2 : 2 + h, 2 : 2 + w]

    # 8 circular neighbors at R=2 (angles 0, pi/4, ..., 7pi/4)
    offsets = [
        (0.0, 2.0), (-1.4142135, 1.4142135), (-2.0, 0.0), (-1.4142135, -1.4142135),
        (0.0, -2.0), (1.4142135, -1.4142135), (2.0, 0.0), (1.4142135, 1.4142135),
    ]
    code_map = np.zeros((h, w), dtype=np.int32)
    for p, (dy, dx) in enumerate(offsets):
        y0, x0 = int(np.floor(dy)), int(np.floor(dx))
        fy, fx = dy - y0, dx - x0
        n00 = padded[2 + y0 : 2 + y0 + h, 2 + x0 : 2 + x0 + w]
        n01 = padded[2 + y0 : 2 + y0 + h, 2 + x0 + 1 : 2 + x0 + 1 + w] if fx > 0 else n00
        n10 = padded[2 + y0 + 1 : 2 + y0 + 1 + h, 2 + x0 : 2 + x0 + w] if fy > 0 else n00
        n11 = padded[2 + y0 + 1 : 2 + y0 + 1 + h, 2 + x0 + 1 : 2 + x0 + 1 + w] if (fx > 0 and fy > 0) else n00
        neighbor = (1 - fy) * ((1 - fx) * n00 + fx * n01) + fy * ((1 - fx) * n10 + fx * n11)
        code_map |= ((neighbor >= center - 1e-5).astype(np.int32) << p)

    return lut[code_map].astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 1. HOG Extractor (576D)
# ─────────────────────────────────────────────────────────────
class HOGExtractor(BaseFeatureExtractor):
    """
    HOG Feature Extractor:
    Sobel gradients, 8x8 pixel cells, 2x2 cell blocks, 9 bins (0-180 deg).
    Output shape: (N, 576) float32.
    """

    def __init__(self):
        self._fitted = False
        self.hog_desc = None

    def _get_hog(self):
        if self.hog_desc is None:
            try:
                importlib.import_module("cv2.objdetect")
            except ImportError:
                pass
            hog_cls = getattr(cv2, "HOGDescriptor", None)
            if hog_cls is None:
                objdetect_mod = getattr(cv2, "objdetect", None)
                if objdetect_mod is not None:
                    hog_cls = getattr(objdetect_mod, "HOGDescriptor", None)

            if hog_cls is not None:
                try:
                    self.hog_desc = hog_cls(
                        _winSize=(64, 64),
                        _blockSize=(16, 16),
                        _blockStride=(16, 16),
                        _cellSize=(8, 8),
                        _nbins=9,
                    )
                except Exception:
                    self.hog_desc = None
        return self.hog_desc

    def _compute_numpy_hog(self, img_64x64: np.ndarray) -> np.ndarray:
        """
        Pure NumPy/Sobel HOG fallback (Task 1.2 mathematical specification):
        Sobel filters, 8x8 pixel cells, 9-bin histograms (0-180 deg),
        2x2 cell blocks with 16x16 block stride (16 blocks total) and L2-Hys normalization.
        Output: 16 * 4 * 9 = 576D float32 vector.
        """
        gx = cv2.Sobel(img_64x64, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img_64x64, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        angle = np.mod(angle, 180.0)

        cell_hist = np.zeros((8, 8, 9), dtype=np.float32)
        bin_width = 20.0
        for cy in range(8):
            for cx in range(8):
                c_mag = mag[cy * 8 : (cy + 1) * 8, cx * 8 : (cx + 1) * 8]
                c_ang = angle[cy * 8 : (cy + 1) * 8, cx * 8 : (cx + 1) * 8]
                bin_idx = np.clip((c_ang / bin_width).astype(int), 0, 8)
                for b in range(9):
                    cell_hist[cy, cx, b] = np.sum(c_mag[bin_idx == b])

        blocks = []
        for by in range(4):
            for bx in range(4):
                block = cell_hist[by * 2 : (by + 1) * 2, bx * 2 : (bx + 1) * 2, :].flatten()
                norm = float(np.linalg.norm(block)) + 1e-6
                block = block / norm
                block = np.clip(block, 0.0, 0.2)
                block = block / (float(np.linalg.norm(block)) + 1e-6)
                blocks.append(block)

        return np.concatenate(blocks).astype(np.float32)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["hog_desc"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.hog_desc = None

    def fit(self, samples: list) -> "HOGExtractor":
        self._fitted = True
        return self

    def _extract_one(self, img: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(img, (64, 64))
        hog_obj = self._get_hog()
        if hog_obj is not None:
            try:
                h = hog_obj.compute(img_resized).flatten()
                return h.astype(np.float32)
            except Exception:
                pass
        return self._compute_numpy_hog(img_resized)

    def transform(self, samples: list) -> np.ndarray:
        return np.array([self._extract_one(img) for img in samples], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 2. Uniform LBP Extractor (944D)
# ─────────────────────────────────────────────────────────────
class LBPExtractor(BaseFeatureExtractor):
    """
    Uniform LBP Feature Extractor (non-rotation-invariant uniform patterns):
    R=2, P=8 neighbours -> exactly 59 uniform-pattern bins per block,
    4x4 spatial grid (16 blocks) -> 16 x 59 = 944 dimensions.
    Output shape: (N, 944) float32.

    NOTE: the task text says P=16, but P=16 gives 243 uniform patterns, so
    16 x 243 = 3888 != 944. P=8 is the only setting that yields 59 bins/944D.
    """

    def __init__(self, radius: int = 2, n_points: int = 8, grid: int = 4):
        self.radius = radius
        self.n_points = n_points
        self.grid = grid
        self.n_bins = 59

    def fit(self, samples: list) -> "LBPExtractor":
        return self

    def _extract_one(self, img: np.ndarray) -> np.ndarray:
        if local_binary_pattern is not None:
            lbp_map = local_binary_pattern(img, P=self.n_points, R=self.radius, method="nri_uniform")
        else:
            lbp_map = _numpy_nri_uniform_lbp_p8_r2(img)
        h, w = img.shape[:2]
        ch, cw = h // self.grid, w // self.grid
        hist_all = []

        for r in range(self.grid):
            for c in range(self.grid):
                cell = lbp_map[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw]
                hist, _ = np.histogram(cell, bins=self.n_bins, range=(0, self.n_bins))
                hist = hist.astype(np.float32)
                s = float(hist.sum())
                if s > 0:
                    hist /= s
                hist_all.append(hist)
        return np.concatenate(hist_all).astype(np.float32)

    def transform(self, samples: list) -> np.ndarray:
        return np.array([self._extract_one(img) for img in samples], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 3. PCA Eigenface Extractor (50D)
# ─────────────────────────────────────────────────────────────
class PCAFaceExtractor(BaseFeatureExtractor):
    """
    PCA Eigenface Extractor:
    Retains top 50 orthogonal eigenvectors capturing >85% variance.
    Output shape: (N, 50) float32.
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, whiten=False)

    def fit(self, samples: list) -> "PCAFaceExtractor":
        X = np.array([img.flatten() for img in samples], dtype=np.float32)
        self.pca.fit(X)
        return self

    def transform(self, samples: list) -> np.ndarray:
        X = np.array([img.flatten() for img in samples], dtype=np.float32)
        return self.pca.transform(X).astype(np.float32)


if __name__ == "__main__":
    import glob

    paths = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "processed", "*", "*.png")))
    if not paths:
        print("[ERROR] No processed images found. Run 'python process_all_data.py' first.")
        sys.exit(1)

    imgs = [img for img in (cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths) if img is not None]
    print(f"[DATA] Loaded {len(imgs)} processed 100x100 face images.")

    hog = HOGExtractor().fit(imgs).transform(imgs)
    lbp = LBPExtractor().fit(imgs).transform(imgs)
    pca_ex = PCAFaceExtractor(50).fit(imgs)
    pca = pca_ex.transform(imgs)

    print(f"  HOG : shape={hog.shape}, dtype={hog.dtype}  (expected (N, 576))")
    print(f"  LBP : shape={lbp.shape}, dtype={lbp.dtype}  (expected (N, 944))")
    print(f"  PCA : shape={pca.shape}, dtype={pca.dtype}  (expected (N, 50))")
    print(f"  PCA explained variance: {pca_ex.pca.explained_variance_ratio_.sum() * 100:.1f}%")

    ok = hog.shape[1] == 576 and lbp.shape[1] == 944 and pca.shape[1] == 50
    print("[SUCCESS] All extractors output the correct dimensions." if ok else "[FAILED] Wrong dimensions.")
