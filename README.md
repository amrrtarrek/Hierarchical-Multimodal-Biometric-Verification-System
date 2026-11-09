# Hierarchical Multimodal Biometric Verification System

A 3-stage hierarchical biometric verification system combining **Face**, **Voice**, and **Fingerprint** modalities using classical machine learning and signal/image processing.

---

## Project Folder Structure by Team Member

| Folder | Member | Modality & Tasks |
| :--- | :---: | :--- |
| [`member_1_face/`](./member_1_face) | **Member 1** | **Task 1.1**: Face Preprocessing (`face_preprocessing.py`)<br>**Task 1.2**: HOG (576D), LBP (944D), PCA (50D) Extractors (`face_extractors.py`)<br>**Task 1.3**: Face SVM & Random Forest (`face_classifiers.py`)<br>**Task 1.4**: Live Webcam Stream (`live_face.py`) |
| [`member_2_voice/`](./member_2_voice) | **Member 2** | **Task 2.1**: VAD & Acoustic Conditioning (`voice_preprocessing.py`)<br>**Task 2.2**: MFCC (40D) + ZCR (3D) + RMS (3D) = 46D Extractor (`voice_extractors.py`)<br>**Task 2.3**: Speaker SVM & Logistic Regression (`voice_classifiers.py`)<br>**Task 2.4**: Live Microphone Stream (`live_voice.py`) |
| [`member_3_fingerprint_rbac/`](./member_3_fingerprint_rbac) | **Member 3** | **Task 3.1**: Fingerprint CLAHE + Otsu Preprocessing (`fingerprint_preprocessing.py`)<br>**Task 3.2**: 2D Gabor Filter Bank (64D) & SVM (`fingerprint_extractors.py`, `fingerprint_classifiers.py`)<br>**Task 3.3**: Speech Command RF & RBAC Policy Engine (`command_classifier.py`, `rbac_engine.py`) |
| [`member_4_fusion_eval/`](./member_4_fusion_eval) | **Member 4** | **Task 4.1**: Base Interfaces & Fixtures (`interfaces.py`, `generate_fixtures.py`)<br>**Task 4.2**: Per-Modality Z-Score Early Fusion (686D) + RBF SVM (`early_fusion.py`)<br>**Task 4.3**: 3-Stage Hierarchical Verification Pipeline (`hierarchical_engine.py`)<br>**Task 4.4**: Evaluation Suite — Confusion Matrices, FAR/FRR/EER, ROC/DET (`evaluation.py`) |

---

## Member 4 (`member_4_fusion_eval`) Files

| File | Task | Purpose |
| :--- | :---: | :--- |
| `interfaces.py` | **Task 4.1** | Abstract contracts `BaseFeatureExtractor`, `BaseBiometricModel`, and `DecisionRecord` |
| `generate_fixtures.py` | **Task 4.1** | Generates synthetic fixtures (`data/fixtures/`) and aligns real team features (`data/processed/fusion/`) |
| `early_fusion.py` | **Task 4.2** | Independent per-modality `StandardScaler` (`576D + 46D + 64D = 686D`) + Platt-calibrated RBF SVM |
| `hierarchical_engine.py` | **Task 4.3** | 3-Stage progressive pipeline (Stage 1 Face Screening $\rightarrow$ Stage 2 Voice & RBAC $\rightarrow$ Stage 3 Multimodal Fusion) |
| `evaluation.py` | **Task 4.4** | Computes Confusion Matrices, $\text{FAR}(\tau)$, $\text{FRR}(\tau)$, $\text{EER}(\tau^*)$, ROC/DET curves, and Markdown report in `reports/figures/` |
| `tests/test_member4_pipeline.py` | **Tests** | Unit & integration tests covering Tasks 4.1 – 4.4 and full team integration |

---

## Quick Commands

```bash
# 1. Generate synthetic & real aligned fixtures (Task 4.1)
python generate_fixtures.py

# 2. Run Early Fusion, Hierarchical Pipeline & Evaluation Suite (Tasks 4.2 - 4.4)
python evaluation.py

# 3. Run Member 4 tests (or full project test suite)
python -m pytest tests/test_member4_pipeline.py
python -m pytest tests/
```
