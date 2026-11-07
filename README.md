# Hierarchical Multimodal Biometric Verification System

A 3-stage hierarchical biometric verification system combining **Face**, **Voice**, and **Fingerprint** modalities using classical machine learning and signal/image processing.

## Team Branches & Workstreams

| Member | Branch | Modality & Scope |
| :--- | :--- | :--- |
| **Member 1** | `feature/face` | Face Preprocessing (Haar Cascade), HOG / LBP / PCA Extractors, Face SVM & RF, Live Webcam Stream |
| **Member 2** | `feature/voice` | Acoustic Conditioning (VAD), MFCC / ZCR / RMS Extractors, Speaker SVM & LR, Live Microphone Stream |
| **Member 3** | `feature/fingerprint-command` | Fingerprint Preprocessing (CLAHE + Otsu), 2D Gabor Filter Bank & SVM, Speech Command RBAC Engine |
| **Member 4** | `feature/fusion-eval` | Base Interfaces, Synthetic Fixtures, Early Feature Fusion (Z-Score + SVM), 3-Stage Hierarchical Pipeline, Evaluation Suite |

---

## Member 4 (`feature/fusion-eval`) Structure

| File | Task | Purpose |
| :--- | :---: | :--- |
| `interfaces.py` | **Task 4.1** | Abstract contracts `BaseFeatureExtractor`, `BaseBiometricModel`, and `DecisionRecord` |
| `generate_fixtures.py` | **Task 4.1** | Generates synthetic fixtures in `data/fixtures/` (`X_face_synthetic.npy`, `X_voice_synthetic.npy`, `X_print_synthetic.npy`, `y_synthetic.npy`) |
| `early_fusion.py` | **Task 4.2** | Independent per-modality `StandardScaler` (`576D + 46D + 64D = 686D`) + Platt-calibrated RBF SVM |
| `hierarchical_engine.py` | **Task 4.3** | 3-Stage progressive pipeline (Stage 1 Face Screening $\rightarrow$ Stage 2 Voice & RBAC $\rightarrow$ Stage 3 Multimodal Fusion) |
| `evaluation.py` | **Task 4.4** | Computes Confusion Matrices, $\text{FAR}(\tau)$, $\text{FRR}(\tau)$, $\text{EER}(\tau^*)$, ROC/DET curves, and Markdown report in `reports/figures/` |
| `tests/test_member4_pipeline.py` | **Tests** | Unit & integration tests covering Tasks 4.1 – 4.4 |

---

## Running Member 4 Pipeline & Tests

```bash
# 1. Generate synthetic fixtures (Task 4.1)
python generate_fixtures.py

# 2. Run Early Fusion, Hierarchical Pipeline & Evaluation Suite (Tasks 4.2 - 4.4)
python evaluation.py

# 3. Run unit tests
python -m pytest
```
