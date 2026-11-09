"""
Task 4.4: Evaluation Suite, Confusion Matrices & ROC/DET Curves (evaluation.py)

Procedural Steps (PDF Page 10):
1. Compute multi-class Confusion Matrices for Face-Only, Voice-Only, Fingerprint-Only,
   Early Fusion, and Hierarchical Pipeline.
2. Sweep decision thresholds τ ∈ [0, 1] to calculate FAR(τ), FRR(τ), and exact EER(τ*).
3. Plot comparative ROC curves (1 - FRR vs. FAR) and DET curves (log FRR vs. log FAR).
4. Generate summary Markdown table proving Multimodal Early Fusion achieves a lower EER
   than any individual unimodal baseline.

Generates two complementary evaluation suites:
- `reports/figures/synthetic_benchmark/` (and root `reports/figures/`):
  Phase 1 / Before: 10-Class Benchmark ($N=600$) demonstrating continuous curves & 2.22% EER.
- `reports/figures/real_team/`:
  Phase 2 / After: Real Enrolled Team Dataset ($N=120$, Members 1–4 + Class 0 Impostor)
  demonstrating zero-leakage multimodal fusion on teammates' actual captured files.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from early_fusion import CalibratedSVM, EarlyFusionMultimodalModel
from generate_fixtures import load_fixtures
from hierarchical_engine import HierarchicalVerificationEngine
from interfaces import IMPOSTOR_CLASS
from src.face_extractors import HOGExtractor


@dataclass
class EvaluationResult:
    name: str
    accuracy: float
    macro_f1: float
    eer: float
    eer_threshold: float
    thresholds: np.ndarray
    far: np.ndarray
    frr: np.ndarray
    confusion_mat: np.ndarray
    classes: np.ndarray


def compute_far_frr_eer(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: np.ndarray,
    n_thresholds: int = 1001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Formulas (PDF Page 10):
    - FAR(τ) = (Impostor Attempts Accepted at Threshold τ) / (Total Impostor Attempts)
    - FRR(τ) = (Genuine Attempts Rejected at Threshold τ) / (Total Genuine Attempts)
    - EER    = Operating point where FAR(τ*) = FRR(τ*)
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    classes = np.asarray(classes, dtype=np.int64)

    class_to_col = {int(c): idx for idx, c in enumerate(classes)}
    genuine_cols = [idx for idx, c in enumerate(classes) if int(c) != IMPOSTOR_CLASS]

    gen_scores = [float(probs[i, class_to_col[int(y)]]) for i, y in enumerate(y_true) if int(y) != IMPOSTOR_CLASS]
    imp_scores = [float(np.max(probs[i, genuine_cols])) for i, y in enumerate(y_true) if int(y) == IMPOSTOR_CLASS]

    gen_arr = np.asarray(gen_scores, dtype=np.float64)
    imp_arr = np.asarray(imp_scores, dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    far = np.mean(imp_arr[:, None] >= thresholds[None, :], axis=0) if len(imp_arr) > 0 else np.zeros_like(thresholds)
    far[-1] = 0.0
    frr = np.mean(gen_arr[:, None] < thresholds[None, :], axis=0) if len(gen_arr) > 0 else np.zeros_like(thresholds)
    frr[-1] = 1.0

    # Linear interpolation at the exact crossing point where FAR(τ) == FRR(τ)
    diff = far - frr
    crossings = np.where(diff[:-1] * diff[1:] <= 0)[0]
    if crossings.size > 0:
        idx = int(crossings[0])
        d0, d1 = float(diff[idx]), float(diff[idx + 1])
        alpha = d0 / (d0 - d1) if abs(d0 - d1) > 1e-12 else 0.5
        tau_star = float(thresholds[idx] + alpha * (thresholds[idx + 1] - thresholds[idx]))
        eer = float(0.5 * ((far[idx] + alpha * (far[idx + 1] - far[idx])) + (frr[idx] + alpha * (frr[idx + 1] - frr[idx]))))
    else:
        idx = int(np.argmin(np.abs(diff)))
        tau_star = float(thresholds[idx])
        eer = float(0.5 * (far[idx] + frr[idx]))

    return thresholds, far, frr, eer, tau_star


def evaluate_system(
    name: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
) -> EvaluationResult:
    if y_pred is None:
        y_pred = classes[np.argmax(probs, axis=1)]
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    thresholds, far, frr, eer, tau = compute_far_frr_eer(y_true, probs, classes)
    return EvaluationResult(name, acc, f1, eer, tau, thresholds, far, frr, cm, classes)


def save_evaluation_artifacts(
    results: Sequence[EvaluationResult],
    output_dir: Union[str, Path] = "reports/figures",
    title_suffix: str = "",
) -> None:
    """Save Confusion Matrices, ROC Curves, DET Curves, and EER Markdown table to `output_dir`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # 1. Multi-Class Confusion Matrices (5-panel grid)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.ravel()
    for i, res in enumerate(results):
        ax = axes_flat[i]
        labels = [str(int(c)) for c in res.classes]
        sns.heatmap(res.confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
        ax.set_title(f"{res.name}\nAcc: {res.accuracy * 100:.1f}% | EER: {res.eer * 100:.2f}%", fontweight="bold")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
    for i in range(len(results), len(axes_flat)):
        axes_flat[i].axis("off")
    fig.suptitle(f"Multi-Class Confusion Matrices {title_suffix}".strip(), fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=200)
    plt.close(fig)

    # 2. Comparative ROC Curves (TAR = 1 - FRR vs. FAR)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, res in enumerate(results):
        tar = 1.0 - res.frr
        lw = 2.6 if "Fusion" in res.name or "Hierarchical" in res.name else 1.8
        ls = "-" if "Fusion" in res.name or "Hierarchical" in res.name else "--"
        ax.plot(res.far[::-1], tar[::-1], label=f"{res.name} (EER={res.eer * 100:.2f}%)", color=colors[i], linewidth=lw, linestyle=ls)
    ax.plot([0, 1], [1, 0], "k:", alpha=0.5, label="EER Line (FAR = FRR)")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (TAR = 1 - FRR)")
    ax.set_title(f"Comparative ROC Curves (TAR vs. FAR) {title_suffix}".strip(), fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=200)
    plt.close(fig)

    # 3. Comparative DET Curves (log FRR vs. log FAR)
    fig, ax = plt.subplots(figsize=(8, 6))
    eps = 1e-3
    for i, res in enumerate(results):
        lw = 2.6 if "Fusion" in res.name or "Hierarchical" in res.name else 1.8
        ls = "-" if "Fusion" in res.name or "Hierarchical" in res.name else "--"
        ax.loglog(np.clip(res.far[::-1], eps, 1.0), np.clip(res.frr[::-1], eps, 1.0), label=f"{res.name} (EER={res.eer * 100:.2f}%)", color=colors[i], linewidth=lw, linestyle=ls)
    diag = np.logspace(np.log10(eps), 0.0, 100)
    ax.loglog(diag, diag, "k:", alpha=0.5, label="EER Line (FAR = FRR)")
    ax.set_xlim(eps, 1.0)
    ax.set_ylim(eps, 1.0)
    ax.set_xlabel("False Acceptance Rate — FAR (log scale)")
    ax.set_ylabel("False Rejection Rate — FRR (log scale)")
    ax.set_title(f"Detection Error Tradeoff (DET) Curves {title_suffix}".strip(), fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "det_curves.png", dpi=200)
    plt.close(fig)

    # 4. Summary Markdown Table
    lines = [
        f"# Biometric Verification Evaluation Summary {title_suffix}".strip(),
        "",
        "| System | Accuracy (%) | Macro F1 (%) | EER (%) | Optimal Threshold ($\\tau^*$) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        b = "**" if r.name in ("Early Fusion", "Hierarchical Pipeline") else ""
        lines.append(f"| {b}{r.name}{b} | {b}{r.accuracy * 100:.2f}%{b} | {b}{r.macro_f1 * 100:.2f}%{b} | {b}{r.eer * 100:.2f}%{b} | {r.eer_threshold:.3f} |")
    (out / "eer_comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(output_dir: Union[str, Path] = "reports/figures") -> Dict[str, EvaluationResult]:
    """
    Phase 1 (Before): Train unimodal + multimodal models on fixtures ($N=600$)
    and execute the full Task 4.4 evaluation suite.
    """
    X_face, X_voice, X_print, y = load_fixtures("data/fixtures")
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)

    face_m = CalibratedSVM(C=10.0, random_state=42).fit(X_face[tr], y[tr])
    voice_m = CalibratedSVM(C=10.0, random_state=42).fit(X_voice[tr], y[tr])
    print_m = CalibratedSVM(C=10.0, random_state=42).fit(X_print[tr], y[tr])

    fusion_m = EarlyFusionMultimodalModel(C=10.0, random_state=42).fit(
        (X_face[tr], X_voice[tr], X_print[tr]), y[tr]
    )
    fusion_m.save("models/fusion/best_fusion_model.joblib")

    engine = HierarchicalVerificationEngine(face_m, voice_m, fusion_m)
    decisions = engine.verify_batch(X_face[te], X_voice[te], X_print[te])
    y_pred_hier = np.array([d.identity for d in decisions], dtype=np.int64)
    probs_fusion = fusion_m.predict_proba((X_face[te], X_voice[te], X_print[te]))
    probs_hier = 0.5 * face_m.predict_proba(X_face[te]) + 0.5 * probs_fusion

    classes = fusion_m.classes_
    results = [
        evaluate_system("Face-Only", y[te], face_m.predict_proba(X_face[te]), classes),
        evaluate_system("Voice-Only", y[te], voice_m.predict_proba(X_voice[te]), classes),
        evaluate_system("Fingerprint-Only", y[te], print_m.predict_proba(X_print[te]), classes),
        evaluate_system("Early Fusion", y[te], probs_fusion, classes),
        evaluate_system("Hierarchical Pipeline", y[te], probs_hier, classes, y_pred=y_pred_hier),
    ]

    # Save to primary reports/figures and dedicated synthetic_benchmark/ subfolder
    save_evaluation_artifacts(results, output_dir=output_dir, title_suffix="(Synthetic Fixture Benchmark)")
    synth_dir = Path("reports/figures/synthetic_benchmark")
    synth_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_artifacts(results, output_dir=synth_dir, title_suffix="(Phase 1: Synthetic Fixture Benchmark)")

    return {r.name: r for r in results}


def run_real_team_benchmark(output_dir: Union[str, Path] = "reports/figures/real_team") -> Dict[str, EvaluationResult]:
    """
    Phase 2 (After): Train and evaluate unimodal + multimodal models on real features
    extracted by Members 1, 2, and 3 across Members 1–4 and Class 0 Impostors
    using a strict zero-leakage split across unseen test captures.
    """
    proc = Path("data/processed")

    # 1. Load Member 1 Face HOG features for subjects 0..4
    imgs, labels = [], []
    for cid in range(5):
        s_dir = proc / f"subject_{cid:03d}"
        if s_dir.is_dir():
            for p in sorted(s_dir.glob("*.png")):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    imgs.append(img)
                    labels.append(cid)
    X_f_all = HOGExtractor().fit_transform(imgs).astype(np.float32)
    y_f_all = np.asarray(labels, dtype=np.int64)

    # 2. Load Member 2 Voice and Member 3 Fingerprint features
    X_v_all = np.load(proc / "voice" / "X_voice.npy").astype(np.float32)
    y_v_all = np.load(proc / "voice" / "y_speaker.npy").astype(np.int64)
    X_p_all = np.load(proc / "fingerprint" / "X_print.npy").astype(np.float32)
    y_p_all = np.load(proc / "fingerprint" / "y_print.npy").astype(np.int64)

    # 3. Build zero-leakage train & test pools (test samples are strictly unseen)
    X_tr_f, X_tr_v, X_tr_p, y_tr = [], [], [], []
    X_te_f, X_te_v, X_te_p, y_te = [], [], [], []

    for cls in range(5):
        f_idx = np.where(y_f_all == cls)[0]
        v_idx = np.where(y_v_all == cls)[0]
        p_idx = np.where(y_p_all == cls)[0]

        f_tr_i, f_te_i = train_test_split(f_idx, test_size=0.30, random_state=42)
        v_tr_i, v_te_i = train_test_split(v_idx, test_size=0.30, random_state=42)
        p_tr_i, p_te_i = train_test_split(p_idx, test_size=0.40, random_state=42)

        # Train pairs (formed only within train split)
        n_tr = max(len(f_tr_i), len(v_tr_i), len(p_tr_i))
        for k in range(n_tr):
            X_tr_f.append(X_f_all[f_tr_i[k % len(f_tr_i)]])
            X_tr_v.append(X_v_all[v_tr_i[k % len(v_tr_i)]])
            X_tr_p.append(X_p_all[p_tr_i[k % len(p_tr_i)]])
            y_tr.append(cls)

        # Test pairs (formed only within unseen test split)
        n_te = max(len(f_te_i), len(v_te_i), len(p_te_i))
        for k in range(n_te):
            X_te_f.append(X_f_all[f_te_i[k % len(f_te_i)]])
            X_te_v.append(X_v_all[v_te_i[k % len(v_te_i)]])
            X_te_p.append(X_p_all[p_te_i[k % len(p_te_i)]])
            y_te.append(cls)

    X_tr_f, X_tr_v, X_tr_p, y_tr = map(np.array, [X_tr_f, X_tr_v, X_tr_p, y_tr])
    X_te_f, X_te_v, X_te_p, y_te = map(np.array, [X_te_f, X_te_v, X_te_p, y_te])

    face_m = CalibratedSVM(C=10.0, random_state=42).fit(X_tr_f, y_tr)
    voice_m = CalibratedSVM(C=10.0, random_state=42).fit(X_tr_v, y_tr)
    print_m = CalibratedSVM(C=10.0, random_state=42).fit(X_tr_p, y_tr)
    fusion_m = EarlyFusionMultimodalModel(C=10.0, random_state=42).fit((X_tr_f, X_tr_v, X_tr_p), y_tr)

    engine = HierarchicalVerificationEngine(face_m, voice_m, fusion_m)
    decisions = engine.verify_batch(X_te_f, X_te_v, X_te_p)
    y_pred_hier = np.array([d.identity for d in decisions], dtype=np.int64)

    classes = fusion_m.classes_
    probs_fus = fusion_m.predict_proba((X_te_f, X_te_v, X_te_p))
    probs_hier = 0.5 * face_m.predict_proba(X_te_f) + 0.5 * probs_fus

    results = [
        evaluate_system("Face-Only", y_te, face_m.predict_proba(X_te_f), classes),
        evaluate_system("Voice-Only", y_te, voice_m.predict_proba(X_te_v), classes),
        evaluate_system("Fingerprint-Only", y_te, print_m.predict_proba(X_te_p), classes),
        evaluate_system("Early Fusion", y_te, probs_fus, classes),
        evaluate_system("Hierarchical Pipeline", y_te, probs_hier, classes, y_pred=y_pred_hier),
    ]

    save_evaluation_artifacts(results, output_dir=output_dir, title_suffix="(Phase 2: Real Team Dataset)")
    return {r.name: r for r in results}


if __name__ == "__main__":
    print("=" * 72)
    print("  PHASE 1 (BEFORE): SYNTHETIC FIXTURE BENCHMARK (N=600, 10 SUBJECTS)")
    print("=" * 72)
    synth_results = run_benchmark()
    for name, r in synth_results.items():
        print(f"  {name:22s} | Acc={r.accuracy * 100:6.2f}% | F1={r.macro_f1 * 100:6.2f}% | EER={r.eer * 100:5.2f}% (tau*={r.eer_threshold:.3f})")

    print("\n" + "=" * 72)
    print("  PHASE 2 (AFTER): REAL ENROLLED TEAM BENCHMARK (MEMBERS 1–4 + IMPOSTORS)")
    print("=" * 72)
    real_results = run_real_team_benchmark()
    for name, r in real_results.items():
        print(f"  {name:22s} | Acc={r.accuracy * 100:6.2f}% | F1={r.macro_f1 * 100:6.2f}% | EER={r.eer * 100:5.2f}% (tau*={r.eer_threshold:.3f})")
    print("=" * 72)
