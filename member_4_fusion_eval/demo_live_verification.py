"""
demo_live_verification.py
Interactive Live Testing Console for the Hierarchical Multimodal Biometric Verification System.
Allows Member 4 to demonstrate live webcam, live audio, and end-to-end hierarchical verification.
"""

import os
import sys
import time
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
from interfaces import (
    FACE_DIM,
    VOICE_DIM,
    PRINT_DIM,
    FUSED_DIM,
    STATUS_VERIFIED,
    STATUS_REJECTED_IMPOSTOR,
    STATUS_ACCESS_DENIED,
)
from early_fusion import EarlyFusionMultimodalModel
from hierarchical_engine import HierarchicalVerificationEngine
from src.rbac_engine import RBACEngine


def print_banner():
    print("=" * 75)
    print("   HIERARCHICAL MULTIMODAL BIOMETRIC VERIFICATION SYSTEM")
    print("   Member 4 — Interactive Live Testing & Defense Console")
    print("=" * 75)


def run_live_webcam():
    print("\n[INFO] Starting Live Webcam Face Verification Stream...")
    print("[INFO] Press 'q' inside the OpenCV camera window to stop.\n")
    try:
        from src.live_face import run_live_stream
        run_live_stream()
    except Exception as e:
        print(f"[ERROR] Could not start webcam: {e}")
        print("[HINT] Ensure a webcam is connected and accessible.")


def run_live_microphone():
    print("\n[INFO] Starting Live Microphone Voice Verification Stream...")
    print("[INFO] Speak into the microphone. Press Ctrl+C in console to stop.\n")
    try:
        from src.live_voice import run_live_voice_stream
        run_live_voice_stream()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped live voice stream.")
    except Exception as e:
        print(f"[ERROR] Could not start microphone: {e}")
        print("[HINT] Ensure a microphone is connected and recording permissions are granted.")


def run_multimodal_simulation():
    print("\n" + "-" * 75)
    print("  SIMULATED MULTIMODAL HIERARCHICAL SCENARIO")
    print("-" * 75)

    # 1. Load trained models
    try:
        face_m = joblib.load("models/face/best_face_model.joblib")
        voice_m = joblib.load("models/voice/best_voice_model.joblib")
        fusion_m = EarlyFusionMultimodalModel.load("models/fusion/best_fusion_model.joblib")
    except Exception as e:
        print(f"[ERROR] Loading models failed: {e}")
        return

    rbac = RBACEngine()
    engine = HierarchicalVerificationEngine(
        face_model=face_m,
        voice_model=voice_m,
        fusion_model=fusion_m,
        rbac_engine=rbac,
        tau_1=0.80,
        tau_2=0.85,
    )

    print("\nSelect Subject to Simulate:")
    print("  [1] Member 1 (Role: Admin)")
    print("  [2] Member 2 (Role: Operator)")
    print("  [3] Member 3 (Role: User)")
    print("  [4] Member 4 (Role: User)")
    print("  [0] Class 0 Impostor (Unknown Intruder)")

    subj_choice = input("\nEnter choice (0-4) [default: 1]: ").strip()
    subj_id = int(subj_choice) if subj_choice in ["0", "1", "2", "3", "4"] else 1

    print("\nSelect Voice Command:")
    print("  [1] Report Status (Standard non-sensitive command)")
    print("  [2] Unlock Door (Operator / Admin access)")
    print("  [3] Override System (Restricted ADMIN command)")
    print("  [4] Trigger Lockdown (Restricted ADMIN command)")

    cmd_choice = input("\nEnter choice (1-4) [default: 1]: ").strip()
    cmd_map = {
        "1": "Report Status",
        "2": "Unlock Door",
        "3": "Override",
        "4": "Lockdown",
    }
    command = cmd_map.get(cmd_choice, "Report Status")

    print("\nSelect Simulation Noise Level:")
    print("  [1] Clean Conditions (High Confidence)")
    print("  [2] Degraded Face (Forces Stage 2 Voice check)")
    print("  [3] Degraded Face & Voice (Forces Stage 3 Full Fusion)")

    noise_choice = input("\nEnter choice (1-3) [default: 1]: ").strip()

    # Load real team features if available
    try:
        X_v_all = np.load("data/processed/voice/X_voice.npy")
        y_v_all = np.load("data/processed/voice/y_speaker.npy")
        X_p_all = np.load("data/processed/fingerprint/X_print.npy")
        y_p_all = np.load("data/processed/fingerprint/y_print.npy")

        v_indices = np.where(y_v_all == subj_id)[0]
        p_indices = np.where(y_p_all == subj_id)[0]

        voice_vec = X_v_all[v_indices[0]] if len(v_indices) > 0 else np.zeros(VOICE_DIM)
        print_vec = X_p_all[p_indices[0]] if len(p_indices) > 0 else np.zeros(PRINT_DIM)
    except Exception:
        voice_vec = np.zeros(VOICE_DIM)
        print_vec = np.zeros(PRINT_DIM)

    # Synthetic or dummy face vector matching dimension
    face_vec = np.random.uniform(0.1, 0.9, size=FACE_DIM)

    if noise_choice == "2":
        # Corrupt face
        face_vec = np.random.normal(0, 0.5, size=FACE_DIM)
    elif noise_choice == "3":
        # Corrupt face and voice
        face_vec = np.random.normal(0, 0.5, size=FACE_DIM)
        voice_vec = np.random.normal(0, 0.5, size=VOICE_DIM)

    print(f"\n[EVALUATING] Running Hierarchical Verification for Subject {subj_id}...")
    time.sleep(0.5)

    decision = engine.verify(
        face_sample=face_vec,
        voice_sample=voice_vec,
        print_sample=print_vec,
        command=command,
    )

    print("\n" + "=" * 55)
    print("          VERIFICATION AUDIT RECORD")
    print("=" * 55)
    print(f"  Subject Tested    : Subject {subj_id}")
    print(f"  Voice Command     : '{command}'")
    print(f"  Stage Reached     : Stage {decision.stage} (1=Face, 2=Voice, 3=Full Fusion)")
    print(f"  Identified As     : Subject {decision.identity}")
    print(f"  Confidence Score  : {decision.confidence * 100:.2f}%")
    print(f"  Final Status      : {decision.status}")
    print("=" * 55)

    if decision.status == STATUS_VERIFIED:
        print("[RESULT] Access GRANTED. Credentials verified.")
    elif decision.status == STATUS_ACCESS_DENIED:
        print("[RESULT] Access DENIED (RBAC Violation): User lacks required role permissions for command.")
    else:
        print("[RESULT] Access REJECTED: User classified as an un-enrolled Impostor.")
    print("=" * 55)


def run_evaluation_suite():
    print("\n[INFO] Running Full Dual-Phase Evaluation Benchmark...\n")
    import evaluation
    print("-" * 72)
    print("  PHASE 1 (BEFORE): SYNTHETIC FIXTURE BENCHMARK (N=600)")
    print("-" * 72)
    s_res = evaluation.run_benchmark()
    for name, r in s_res.items():
        print(f"  {name:22s} | Acc={r.accuracy*100:6.2f}% | EER={r.eer*100:5.2f}% (tau*={r.eer_threshold:.3f})")

    print("\n" + "-" * 72)
    print("  PHASE 2 (AFTER): REAL ENROLLED TEAM BENCHMARK (N=120)")
    print("-" * 72)
    r_res = evaluation.run_real_team_benchmark()
    for name, r in r_res.items():
        print(f"  {name:22s} | Acc={r.accuracy*100:6.2f}% | EER={r.eer*100:5.2f}% (tau*={r.eer_threshold:.3f})")


def run_unit_tests():
    print("\n[INFO] Executing Pytest Automated Verification Suite...\n")
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "tests/"])


def main():
    while True:
        print_banner()
        print("  [1] Live Webcam Face Verification Stream (Member 1)")
        print("  [2] Live Microphone Voice Verification Stream (Member 2)")
        print("  [3] Interactive Multimodal Hierarchical Engine Demo (Member 4)")
        print("  [4] Run Dual-Phase Evaluation Suite & Regenerate Figures (Task 4.4)")
        print("  [5] Run All 109 Automated Pytest Tests")
        print("  [0] Exit")
        print("-" * 75)

        choice = input("Enter choice (0-5): ").strip()
        if choice == "1":
            run_live_webcam()
        elif choice == "2":
            run_live_microphone()
        elif choice == "3":
            run_multimodal_simulation()
        elif choice == "4":
            run_evaluation_suite()
        elif choice == "5":
            run_unit_tests()
        elif choice == "0":
            print("\nExiting. Good luck with your defense!")
            break
        else:
            print("\n[WARNING] Invalid choice. Please enter 0-5.")
        input("\nPress Enter to return to main menu...")


if __name__ == "__main__":
    main()
