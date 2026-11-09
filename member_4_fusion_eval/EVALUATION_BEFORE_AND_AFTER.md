# Member 4 Evaluation: Before (Synthetic Benchmark) vs. After (Real Team Data)

This document provides the complete comparison, figure catalog, and oral defense walkthrough for **Member 4** (Multimodal Fusion, Hierarchical Pipeline & Evaluation).

---

## 1. Executive Summary: Why Two Evaluation Phases?

As Member 4, you are responsible for proving two distinct aspects of the multimodal biometric system to your TA:

1. **Phase 1 — Before (Synthetic Fixtures Benchmark, $N=600$, 10 Subjects)**:
   - **Purpose**: Evaluates biometric error dynamics (EER, ROC, DET) across dense score distributions under varying simulated noise levels.
   - **Why Necessary**: Continuous curves and exact Equal Error Rate ($FAR = FRR$) calculations require sufficient genuine and impostor score density (10 subjects $\times$ 60 samples = 600 total samples; 180 test trials with 150 impostor tests).
   - **Result**: Proves **Early Fusion significantly outperforms all unimodal classifiers**, dropping EER down to **$2.22\%$** (compared to Fingerprint $15.56\%$, Voice $5.19\%$, Face $6.67\%$).

2. **Phase 2 — After (Real Team Dataset, $N=120$, Members 1–4 + Class 0 Impostors)**:
   - **Purpose**: Evaluates the trained unimodal and multimodal models on the actual physical data captured by your teammates (Members 1, 2, and 3) using a strict zero-leakage split across unseen test captures.
   - **Why Necessary**: Proves real-world hardware integration and actual biometric enrollment without data leakage.
   - **Result**: Confirms **100% verification accuracy and 0.00% impostor leakage** for Early Fusion on enrolled team members.

---

## 2. Direct Side-by-Side Metric Comparison

| Model / Pipeline | Phase 1 (Synthetic) Acc | Phase 1 EER | Phase 1 $\tau^*$ | Phase 2 (Real Team) Acc | Phase 2 EER | Phase 2 $\tau^*$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Face-Only** | 93.33% | 6.67% | 0.235 | 91.67% | 0.00% | 0.084 |
| **Voice-Only** | 91.11% | 5.19% | 0.185 | 94.44% | 0.00% | 0.090 |
| **Fingerprint-Only** | 87.22% | 15.56% | 0.320 | 100.00% | 0.00% | 0.092 |
| **Early Fusion (SVM)** | **97.22%** | **2.22%** | **0.284** | **100.00%** | **0.00%** | **0.070** |
| **Hierarchical Pipeline** | 92.78% | 6.67% | 0.310 | 88.89% | 0.00% | 0.074 |

---

## 3. Figure Catalog and File Locations

All evaluation figures have been placed in clean, isolated folders inside `member_4_fusion_eval/figures/`:

### A. Phase 1 — Before (Synthetic Fixtures Benchmark)
Located in: [`member_4_fusion_eval/figures/before_synthetic_benchmark/`](figures/before_synthetic_benchmark/)

- **Confusion Matrices**: [`confusion_matrices.png`](figures/before_synthetic_benchmark/confusion_matrices.png)
  - Displays $10 \times 10$ matrices showing genuine identification and impostor rejection (Class 0).
- **ROC Curves**: [`roc_curves.png`](figures/before_synthetic_benchmark/roc_curves.png)
  - Shows smooth True Positive Rate vs. False Positive Rate curves; Early Fusion achieves $AUC = 0.99$.
- **DET Curves**: [`det_curves.png`](figures/before_synthetic_benchmark/det_curves.png)
  - Shows $\log(FRR)$ vs. $\log(FAR)$ with clear intersections along the $FAR=FRR$ diagonal line.
- **Summary Table**: [`eer_comparison_table.md`](figures/before_synthetic_benchmark/eer_comparison_table.md)

### B. Phase 2 — After (Real Enrolled Team Dataset)
Located in: [`member_4_fusion_eval/figures/after_real_team/`](figures/after_real_team/)

- **Confusion Matrices**: [`confusion_matrices.png`](figures/after_real_team/confusion_matrices.png)
  - Displays $5 \times 5$ matrices for Subjects 0 (Impostor), 1, 2, 3, and 4 (Enrolled Team).
- **ROC Curves**: [`roc_curves.png`](figures/after_real_team/roc_curves.png)
  - Shows step-response performance on unseen real data splits; $AUC = 1.00$ for Early Fusion.
- **DET Curves**: [`det_curves.png`](figures/after_real_team/det_curves.png)
  - Illustrates the discrete step tradeoff curves on small-batch team samples.
- **Summary Table**: [`eer_comparison_table.md`](figures/after_real_team/eer_comparison_table.md)

---

## 4. TA Defense Talking Points for Member 4

When presenting your work to the TA, explain your design and evaluation with these three concise steps:

### Q1: "Why do you have two sets of figures (synthetic vs. real team data)?"
> **Your Answer**:
> *"In biometric verification, calculating a smooth, continuous Equal Error Rate (EER) and drawing publication-quality DET curves requires hundreds of genuine and impostor matching scores across multiple subjects. Because our team dataset currently has 4 enrolled members with 5 fingerprint impressions each (20 total impressions), running a 1,001-point threshold sweep on 20 samples causes metric quantization (step functions where EER falls immediately to 0.00%).*
>
> *Therefore, we used a two-phase evaluation:*
> 1. * **Phase 1 (Synthetic Fixtures, $N=600$)**: Evaluates continuous error dynamics across 10 subjects, proving Early Fusion drops EER from 15.56% down to 2.22%.*
> 2. * **Phase 2 (Real Team Data)**: Evaluates the exact real features extracted by Members 1, 2, and 3 across our team and zero-leakage impostors, proving 100% accuracy and zero impostor leakage on real hardware.*"

### Q2: "How does your Early Fusion work and how do you prevent feature scale domination?"
> **Your Answer**:
> *"Each modality has a different dimensionality and feature scale: Face is 576D HOG, Voice is 46D MFCC+chroma, and Fingerprint is 64D Gabor+statistics. To prevent the high-dimensional face feature from dominating voice and fingerprint, `EarlyFusionMultimodalModel` fits three independent `StandardScaler` instances (one per modality) before concatenating them into a 686-dimensional vector. The fused vector is classified using a calibrated RBF SVM."*

### Q3: "How does the Hierarchical Engine operate and what is its computational advantage?"
> **Your Answer**:
> *"The hierarchical engine operates across three sequential stages:*
> 1. * **Stage 1 (Face Verification)**: Checks face score against confidence threshold $\tau_1 = 0.80$. If confident, the user is immediately accepted or rejected without waking up audio or fingerprint hardware (saving latency and power).*
> 2. * **Stage 2 (Voice + Speech Command Verification)**: If face is uncertain, voice is captured and classified. If high confidence ($\tau_2 = 0.85$), the access decision is returned.*
> 3. * **Stage 3 (Full Multimodal Fusion + RBAC Policy)**: If uncertainty remains or if the attempted command is classified as a sensitive administrative command (`lockdown` or `override`), Stage 3 triggers Early Fusion and verifies the subject's role against the RBAC policy before granting access.*"

---

## 5. Verification Commands

To reproduce both Phase 1 and Phase 2 figures anytime, run:
```powershell
python evaluation.py
```

To run all 109 automated tests (including Member 4's 4-stage pipeline tests):
```powershell
python -m pytest tests/
```

