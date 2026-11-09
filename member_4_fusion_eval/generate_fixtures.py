"""
Task 4.1: Synthetic Fixture Matrix Generator (generate_fixtures.py)

Generates synthetic feature matrices matching exact modality dimensionalities to unblock
Early Fusion (Task 4.2), Hierarchical Pipeline (Task 4.3), and Evaluation (Task 4.4)
before Members 1, 2, and 3 finish real feature extraction:
- data/fixtures/X_face_synthetic.npy  : shape (N, 576), float32
- data/fixtures/X_voice_synthetic.npy : shape (N, 46),  float32
- data/fixtures/X_print_synthetic.npy : shape (N, 64),  float32
- data/fixtures/y_synthetic.npy       : shape (N,),     int64 (Class 0: Impostor, Classes 1..10: Genuine)
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from interfaces import FACE_DIM, IMPOSTOR_CLASS, PRINT_DIM, VOICE_DIM


def generate_synthetic_fixtures(
    output_dir: Union[str, Path] = "data/fixtures",
    n_genuine_classes: int = 10,
    samples_per_class: int = 45,
    n_impostor_samples: int = 150,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic feature matrices with realistic modality scale differences:
    - Face (576D)       : small positive HOG scale in [0, 1]
    - Voice (46D)       : large negative/positive MFCC cepstral scale in [-150, +80]
    - Fingerprint (64D) : medium Gabor filter response scale in [5, 95]
    """
    rng = np.random.default_rng(random_state)
    n_total = n_genuine_classes * samples_per_class + n_impostor_samples

    y_genuine = np.repeat(np.arange(1, n_genuine_classes + 1, dtype=np.int64), samples_per_class)
    y_impostor = np.zeros(n_impostor_samples, dtype=np.int64)
    y = np.concatenate([y_genuine, y_impostor])

    # Latent class prototypes projected to each modality's dimension
    latent_dim = 16
    proj_f = rng.normal(0.0, 1.0 / np.sqrt(latent_dim), size=(latent_dim, FACE_DIM))
    proj_v = rng.normal(0.0, 1.0 / np.sqrt(latent_dim), size=(latent_dim, VOICE_DIM))
    proj_p = rng.normal(0.0, 1.0 / np.sqrt(latent_dim), size=(latent_dim, PRINT_DIM))

    face_centers = {c: rng.normal(0.0, 1.15, size=latent_dim) @ proj_f for c in range(1, n_genuine_classes + 1)}
    voice_centers = {c: rng.normal(0.0, 1.15, size=latent_dim) @ proj_v for c in range(1, n_genuine_classes + 1)}
    print_centers = {c: rng.normal(0.0, 1.18, size=latent_dim) @ proj_p for c in range(1, n_genuine_classes + 1)}

    n_imp_clusters = 15
    imp_f = [rng.normal(0.0, 1.25, size=latent_dim) @ proj_f for _ in range(n_imp_clusters)]
    imp_v = [rng.normal(0.0, 1.25, size=latent_dim) @ proj_v for _ in range(n_imp_clusters)]
    imp_p = [rng.normal(0.0, 1.25, size=latent_dim) @ proj_p for _ in range(n_imp_clusters)]

    X_face = np.zeros((n_total, FACE_DIM), dtype=np.float64)
    X_voice = np.zeros((n_total, VOICE_DIM), dtype=np.float64)
    X_print = np.zeros((n_total, PRINT_DIM), dtype=np.float64)

    for i, cls in enumerate(y):
        if cls != IMPOSTOR_CLASS:
            fp, vp, pp = face_centers[int(cls)], voice_centers[int(cls)], print_centers[int(cls)]
            uf, uv, up = rng.random(3)
            # Clean (~70%) vs degraded (~30%) samples to trigger Stage 1 -> Stage 2 -> Stage 3 escalations
            fn = 0.45 if uf < 0.68 else (0.95 if uf < 0.92 else 1.20)
            vn = 0.40 if uv < 0.72 else (0.90 if uv < 0.92 else 1.20)
            pn = 0.42 if up < 0.74 else (0.92 if up < 0.93 else 1.20)
            if uf >= 0.68:
                fp = (0.62 if uf < 0.92 else 0.42) * fp
            if uv >= 0.72:
                vp = (0.58 if uv < 0.92 else 0.35) * vp
            if up >= 0.74:
                pp = (0.60 if up < 0.93 else 0.38) * pp
        else:
            cid = i % n_imp_clusters
            fp, vp, pp = imp_f[cid], imp_v[(cid + 2) % n_imp_clusters], imp_p[(cid + 5) % n_imp_clusters]
            # Occasional single-modality resemblance to an enrolled subject (rejected by Multimodal Fusion)
            roll = rng.random()
            target = (i % n_genuine_classes) + 1
            if roll < 0.10:
                fp = 0.72 * face_centers[target] + 0.28 * fp
            elif roll < 0.20:
                vp = 0.72 * voice_centers[target] + 0.28 * vp
            elif roll < 0.30:
                pp = 0.72 * print_centers[target] + 0.28 * pp
            fn = vn = pn = 0.72

        X_face[i] = fp + rng.normal(0.0, fn, size=FACE_DIM)
        X_voice[i] = vp + rng.normal(0.0, vn, size=VOICE_DIM)
        X_print[i] = pp + rng.normal(0.0, pn, size=PRINT_DIM)

    # Apply realistic modality scales (proving why per-modality StandardScaler is required in Task 4.2)
    X_face = np.clip(0.18 + 0.075 * X_face, 0.0, 1.0)
    X_voice[:, :20] = -45.0 + 32.0 * X_voice[:, :20]
    X_voice[:, 20:40] = np.clip(18.0 + 6.5 * X_voice[:, 20:40], 0.5, 80.0)
    X_voice[:, 40:] = np.clip(0.10 + 0.03 * X_voice[:, 40:], 0.001, 0.80)
    X_print[:, 0::2] = 42.0 + 14.0 * X_print[:, 0::2]
    X_print[:, 1::2] = np.clip(19.0 + 5.5 * X_print[:, 1::2], 0.5, 95.0)

    perm = rng.permutation(n_total)
    X_face = X_face[perm].astype(np.float32)
    X_voice = X_voice[perm].astype(np.float32)
    X_print = X_print[perm].astype(np.float32)
    y = y[perm].astype(np.int64)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "X_face_synthetic.npy", X_face)
    np.save(out_path / "X_voice_synthetic.npy", X_voice)
    np.save(out_path / "X_print_synthetic.npy", X_print)
    np.save(out_path / "y_synthetic.npy", y)

    return {"X_face": X_face, "X_voice": X_voice, "X_print": X_print, "y": y}


def load_fixtures(
    fixtures_dir: Union[str, Path] = "data/fixtures",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load synthetic fixtures from disk (generating them automatically if not yet present)."""
    base = Path(fixtures_dir)
    files = [
        base / "X_face_synthetic.npy",
        base / "X_voice_synthetic.npy",
        base / "X_print_synthetic.npy",
        base / "y_synthetic.npy",
    ]
    if not all(f.exists() for f in files):
        data = generate_synthetic_fixtures(output_dir=base)
        return data["X_face"], data["X_voice"], data["X_print"], data["y"]

    return (
        np.load(files[0]).astype(np.float32),
        np.load(files[1]).astype(np.float32),
        np.load(files[2]).astype(np.float32),
        np.load(files[3]).astype(np.int64),
    )


if __name__ == "__main__":
    fixtures = generate_synthetic_fixtures()
    for name, arr in fixtures.items():
        print(f"{name:8s}: shape={arr.shape}, dtype={arr.dtype}")
