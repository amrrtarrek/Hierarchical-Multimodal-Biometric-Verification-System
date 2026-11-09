"""
src/rbac_engine.py
Task 3.3 (Steps 2-4): Identity-Governed Role-Based Access Control (RBAC) Policy Engine
& Live Microphone Audio Stream Integration (Day 4 & 5 | Bonus 1).

RBAC Permission Policy (PDF Page 8):
- Member 1 (Admin):    Authorized for all commands ("Unlock System", "Report Status", "Terminate Session").
- Member 2 (Operator): Authorized for "Report Status" and "Unlock System" only.
- Member 3 (Auditor):  Authorized for "Report Status" only.
- Member 4 (Auditor):  Authorized for "Report Status" only.
- Class 0 (Impostor):  Access denied on all commands.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import queue
import sys
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np

from src.command_classifier import COMMANDS, CommandClassifier
from src.voice_extractors import VoiceFeatureExtractor

UNLOCK, STATUS, TERMINATE = COMMANDS
IMPOSTOR: int = 0

ROLES: Dict[int, str] = {
    1: "Admin",
    2: "Operator",
    3: "Auditor",
    4: "Auditor",
}

POLICY: Dict[str, Set[str]] = {
    "Admin": {UNLOCK, STATUS, TERMINATE},
    "Operator": {UNLOCK, STATUS},
    "Auditor": {STATUS},
}

RBAC_PERMISSIONS: Dict[int, Set[str]] = {
    0: set(),
    1: {UNLOCK, STATUS, TERMINATE},
    2: {UNLOCK, STATUS},
    3: {STATUS},
    4: {STATUS},
}


class PermissionDenied(Exception):
    """Raised when an unverified/Class 0 user or restricted role attempts an unauthorized command."""


@dataclass
class Decision:
    allowed: bool
    identity: int
    role: str
    command: str
    reason: str


RBACDecision = Decision


def check_rbac_permission(identity: int, command: str) -> bool:
    """Return True if `identity` is authorized to execute `command` under the RBAC policy."""
    ident = int(identity)
    if ident == IMPOSTOR or ident not in ROLES:
        return False
    role = ROLES[ident]
    return str(command).strip() in POLICY.get(role, set())


class RBACEngine:
    """
    Task 3.3 Steps 2 & 3: Identity-Governed RBAC Policy Engine.
    Executes valid actions for authorized members and raises `PermissionDenied`
    on unauthorized commands, restricted privilege escalations (e.g. "Terminate Session"),
    or Class 0 impostor attempts.
    """

    def __init__(
        self,
        handlers: Optional[Dict[str, Callable[[], str]]] = None,
        min_command_conf: float = 0.50,
    ) -> None:
        self.min_command_conf = float(min_command_conf)
        self.handlers: Dict[str, Callable[[], str]] = handlers or {
            UNLOCK: lambda: "System unlocked.",
            STATUS: lambda: "Status: all systems nominal.",
            TERMINATE: lambda: "Session terminated.",
        }

    def check(
        self,
        identity: int,
        command: str,
        command_conf: float = 1.0,
    ) -> Decision:
        ident = int(identity)
        cmd = str(command).strip()
        role = ROLES.get(ident, "None")

        if ident == IMPOSTOR or ident not in ROLES:
            return Decision(
                allowed=False,
                identity=ident,
                role=role,
                command=cmd,
                reason="Unverified/impostor: denied on all commands",
            )

        if cmd not in COMMANDS:
            return Decision(
                allowed=False,
                identity=ident,
                role=role,
                command=cmd,
                reason=f"Unknown command '{cmd}'",
            )

        if float(command_conf) < self.min_command_conf:
            return Decision(
                allowed=False,
                identity=ident,
                role=role,
                command=cmd,
                reason=f"Command confidence too low ({command_conf:.2f} < {self.min_command_conf:.2f})",
            )

        if cmd not in POLICY[role]:
            return Decision(
                allowed=False,
                identity=ident,
                role=role,
                command=cmd,
                reason=f"{role} (Member {ident}) is not authorized for '{cmd}'",
            )

        return Decision(
            allowed=True,
            identity=ident,
            role=role,
            command=cmd,
            reason="Authorized",
        )

    def execute(
        self,
        identity: int,
        command: str,
        command_conf: float = 1.0,
    ) -> str:
        decision = self.check(identity, command, command_conf)
        if not decision.allowed:
            raise PermissionDenied(f"PERMISSION DENIED: {decision.reason}")
        return self.handlers[decision.command]()


class LiveCommandListener:
    """
    Task 3.3 Step 4: Connects the Speech Command Recognizer and RBAC Policy Engine
    to Member 2's live microphone audio stream (`src/live_voice.py` / `ACTIVE_AUDIO_BUFFER_QUEUE`).

    Supports both:
    - Dictionary items pushed by `LiveVoiceVerificationStream` (`{"speaker_id": ..., "features_46d": ..., "audio": ...}`)
    - 2-tuples `(speaker_id, audio_buffer_or_46d_features)`
    """

    def __init__(
        self,
        command_model: Optional[CommandClassifier] = None,
        feature_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        engine: Optional[RBACEngine] = None,
    ) -> None:
        if command_model is None:
            for candidate in (
                "models/fingerprint/best_command_model.joblib",
                "models/voice/best_command_model.joblib",
            ):
                if os.path.isfile(candidate):
                    command_model = joblib.load(candidate)
                    break
        self.cmd = command_model
        if feature_fn is None:
            extractor = VoiceFeatureExtractor().fit([])
            self.feature_fn = lambda buf: (
                np.asarray(buf, dtype=np.float32)
                if np.asarray(buf).size == 46
                else extractor.transform(buf)[0]
            )
        else:
            self.feature_fn = feature_fn
        self.engine = engine or RBACEngine()

    def handle(
        self,
        speaker_id_or_record: Union[int, Dict[str, Any], Tuple[int, np.ndarray]],
        audio_buffer: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Classify the spoken command and enforce RBAC execution permission for the speaker.
        """
        if isinstance(speaker_id_or_record, dict):
            speaker_id = int(speaker_id_or_record["speaker_id"])
            if "features_46d" in speaker_id_or_record and speaker_id_or_record["features_46d"] is not None:
                feat_vec = np.asarray(speaker_id_or_record["features_46d"], dtype=np.float32)
            else:
                feat_vec = self.feature_fn(speaker_id_or_record["audio"])
        elif isinstance(speaker_id_or_record, tuple) and len(speaker_id_or_record) == 2:
            speaker_id = int(speaker_id_or_record[0])
            feat_vec = self.feature_fn(speaker_id_or_record[1])
        else:
            speaker_id = int(speaker_id_or_record)
            if audio_buffer is None:
                raise ValueError("audio_buffer must be provided when speaker_id is passed as an int.")
            feat_vec = self.feature_fn(audio_buffer)

        command, conf = self.cmd.predict_with_confidence(feat_vec)
        role = ROLES.get(speaker_id, "Impostor" if speaker_id == 0 else "Unknown")

        try:
            msg = self.engine.execute(speaker_id, command, conf)
            return {
                "status": "EXECUTED",
                "identity": speaker_id,
                "role": role,
                "command": command,
                "confidence": conf,
                "result": msg,
            }
        except PermissionDenied as e:
            return {
                "status": "PERMISSION_DENIED",
                "identity": speaker_id,
                "role": role,
                "command": command,
                "confidence": conf,
                "result": str(e),
            }

    def run(self, audio_queue: "queue.Queue", stop_event: Any) -> None:
        """Consume utterances from Member 2's active audio buffer queue until `stop_event` is set."""
        while not stop_event.is_set():
            try:
                item = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(item, tuple) and len(item) == 2:
                res = self.handle(item[0], item[1])
            else:
                res = self.handle(item)
            print(res, flush=True)

