# Hierarchical Multimodal Biometric Verification System

A 3-stage hierarchical biometric verification system combining **Face**, **Voice**, and **Fingerprint** modalities using classical machine learning and signal/image processing.

## Team Branches & Workstreams

| Member | Branch | Modality & Scope |
| :--- | :--- | :--- |
| **Member 1** | `feature/face` | Face Preprocessing (Haar Cascade), HOG / LBP / PCA Extractors, Face SVM & RF, Live Webcam Stream |
| **Member 2** | `feature/voice` | Acoustic Conditioning (VAD), MFCC / ZCR / RMS Extractors, Speaker SVM & LR, Live Microphone Stream |
| **Member 3** | `feature/fingerprint-command` | Fingerprint Preprocessing (CLAHE + Otsu), 2D Gabor Filter Bank & SVM, Speech Command RBAC Engine |
| **Member 4** | `feature/fusion-eval` | Base Interfaces, Synthetic Fixtures, Early Feature Fusion (Z-Score + SVM), 3-Stage Hierarchical Pipeline, Evaluation Suite |

## Getting Started for Teammates

```bash
# 1. Clone the repository
git clone https://github.com/amrrtarrek/Hierarchical-Multimodal-Biometric-Verification-System.git
cd Hierarchical-Multimodal-Biometric-Verification-System

# 2. Switch to your assigned feature branch
git checkout feature/face                  # Member 1
git checkout feature/voice                 # Member 2
git checkout feature/fingerprint-command   # Member 3
git checkout feature/fusion-eval           # Member 4
```
