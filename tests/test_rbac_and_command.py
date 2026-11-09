"""
tests/test_rbac_and_command.py
Unit tests for Task 3.3: Isolated-Word Speech Command Recognizer & RBAC Policy Engine.
"""
import os
import queue
import threading
import joblib
import numpy as np
import pytest

from src.base_interfaces import BaseBiometricModel
from src.command_classifier import (
    COMMANDS,
    CommandClassifier,
    load_command_training_dataset,
    speaker_invariance_cv,
    train_and_save_command_model,
)
from src.live_voice import LiveVoiceVerificationStream
from src.rbac_engine import (
    LiveCommandListener,
    PermissionDenied,
    RBACEngine,
    check_rbac_permission,
)
from src.voice_preprocessing import load_conditioned_wav


def make_synthetic_command_data():
    rng = np.random.default_rng(2)
    y = np.tile([0, 1, 2], 40)
    spk = np.repeat(np.arange(8), 15)
    X = rng.normal(size=(120, 46)).astype(np.float32)
    X[:, :3] += np.eye(3, dtype=np.float32)[y] * 4.5
    return X, y, spk


class TestTask33CommandRecognizerAndRBAC:
    def test_command_classifier_inherits_base_biometric_model(self):
        assert isinstance(CommandClassifier(), BaseBiometricModel)

    def test_command_classifier_speaker_invariance_and_string_predictions(self):
        """Step 1: Speaker-invariant RF command classification returning target word labels."""
        X, y, spk = make_synthetic_command_data()
        assert speaker_invariance_cv(X, y, spk) > 0.90

        clf = CommandClassifier().fit(X, y)
        pred_strs = clf.predict(X[:3])
        pred_ids = clf.predict_ids(X[:3])
        proba = clf.predict_proba(X[:3])

        assert pred_strs.tolist() == [COMMANDS[i] for i in y[:3]]
        assert pred_ids.tolist() == y[:3].tolist()
        assert proba.shape == (3, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.parametrize(
        "ident,cmd,allowed",
        [
            # Member 1 (Admin): Authorized for all 3 commands
            (1, "Unlock System", True),
            (1, "Report Status", True),
            (1, "Terminate Session", True),
            # Member 2 (Operator): Authorized for Report Status & Unlock System only
            (2, "Unlock System", True),
            (2, "Report Status", True),
            (2, "Terminate Session", False),
            # Member 3 (Auditor): Authorized for Report Status only
            (3, "Report Status", True),
            (3, "Unlock System", False),
            (3, "Terminate Session", False),
            # Member 4 (Auditor): Authorized for Report Status only
            (4, "Report Status", True),
            (4, "Unlock System", False),
            (4, "Terminate Session", False),
            # Class 0 (Impostor) & Unknown identities: Denied on all commands
            (0, "Report Status", False),
            (0, "Unlock System", False),
            (0, "Terminate Session", False),
            (99, "Report Status", False),
        ],
    )
    def test_rbac_permission_matrix_and_strict_denial(self, ident, cmd, allowed):
        """Steps 2 & 3 + Definition of Done: Execute authorized actions and throw PermissionDenied otherwise."""
        engine = RBACEngine()
        assert check_rbac_permission(ident, cmd) is allowed

        if allowed:
            result = engine.execute(ident, cmd)
            assert isinstance(result, str) and len(result) > 0
        else:
            with pytest.raises(PermissionDenied):
                engine.execute(ident, cmd)

    def test_low_command_confidence_denied(self):
        with pytest.raises(PermissionDenied):
            RBACEngine(min_command_conf=0.50).execute(1, "Unlock System", command_conf=0.25)

    def test_live_command_listener_with_member2_microphone_stream(self):
        """
        Step 4: Connect command model to Member 2's live microphone audio stream queue
        and verify live voice-command execution and privilege escalation checks.
        """
        model_path = "models/fingerprint/best_command_model.joblib"
        if not os.path.isfile(model_path):
            train_and_save_command_model()

        cmd_model = joblib.load(model_path)
        listener = LiveCommandListener(command_model=cmd_model)

        # 1. Test consuming a rich record from Member 2's LiveVoiceVerificationStream
        shared_q = queue.Queue(maxsize=16)
        voice_stream = LiveVoiceVerificationStream(audio_queue=shared_q)
        X_all = np.load("data/processed/voice/X_voice.npy")
        y_spk = np.load("data/processed/voice/y_speaker.npy")
        y_cmd = np.load("data/processed/voice/y_command.npy")

        # Member 2 (ID=2) saying "Unlock System" (cmd=0)
        m2_unlock_idx = int(np.where((y_spk == 2) & (y_cmd == 0))[0][0])
        m2_unlock_record = {
            "speaker_id": 2,
            "features_46d": X_all[m2_unlock_idx],
            "confidence": 0.98,
            "status": "VERIFIED",
        }
        shared_q.put_nowait(m2_unlock_record)
        queued_record = shared_q.get_nowait()
        res_m2_unlock = listener.handle(queued_record)
        assert res_m2_unlock["identity"] == 2
        assert res_m2_unlock["command"] == "Unlock System"
        assert res_m2_unlock["status"] == "EXECUTED"

        # 2. Test privilege escalation rejection: Member 2 (Operator) attempting "Terminate Session" (cmd=2)
        m2_term_idx = int(np.where((y_spk == 2) & (y_cmd == 2))[0][0])
        rec_term = {
            "speaker_id": 2,
            "features_46d": X_all[m2_term_idx],
            "confidence": 0.98,
            "status": "VERIFIED",
        }
        res_m2_term = listener.handle(rec_term)
        assert res_m2_term["identity"] == 2
        assert res_m2_term["command"] == "Terminate Session"
        assert res_m2_term["status"] == "PERMISSION_DENIED"

        # 3. Test tuple format compatibility (speaker_id, feature_vec)
        X, y, _ = make_synthetic_command_data()
        synth_listener = LiveCommandListener(command_model=CommandClassifier().fit(X, y), feature_fn=lambda b: b)
        term_vec = X[y == 2][0]
        assert synth_listener.handle(3, term_vec)["status"] == "PERMISSION_DENIED"
        assert synth_listener.handle(0, term_vec)["status"] == "PERMISSION_DENIED"
        assert synth_listener.handle(1, term_vec)["status"] == "EXECUTED"
