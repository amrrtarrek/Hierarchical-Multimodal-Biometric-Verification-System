"""
Task 4.3: 3-Stage Hierarchical Verification Pipeline (hierarchical_engine.py)

Procedural Steps (PDF Page 9-10):
1. Stage 1 (Face Screening):
   Extract HOG (576D) -> Face SVM.
   - If P_face >= 0.88, return VERIFIED at Stage 1.
   - If P_face < 0.35, return REJECTED_IMPOSTOR at Stage 1.
   - Else escalate to Stage 2.
2. Stage 2 (Voice & Command):
   Extract MFCC/ZCR/RMS (46D) -> Voice & Command SVM.
   - If Voice Identity == Face Identity and P_voice >= 0.75, evaluate RBAC permission.
     Return VERIFIED or PERMISSION_DENIED at Stage 2.
   - Else escalate to Stage 3.
3. Stage 3 (Full Multimodal Arbitration):
   Extract Fingerprint Gabor (64D) -> Early Fusion (686D) -> Multimodal SVM.
   - If P_fused >= 0.60 and class != 0, return VERIFIED at Stage 3;
   - Else return REJECTED_IMPOSTOR at Stage 3.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import joblib
import numpy as np

from early_fusion import EarlyFeatureFusionExtractor, EarlyFusionMultimodalModel
from interfaces import (
    IMPOSTOR_CLASS,
    STATUS_PERMISSION_DENIED,
    STATUS_REJECTED_IMPOSTOR,
    STATUS_VERIFIED,
    BaseBiometricModel,
    DecisionRecord,
)

# Role-Based Access Control (RBAC) policy table from Task 3.3 (PDF Page 8)
RBAC_PERMISSIONS: Dict[int, Set[str]] = {
    0: set(),                                                         # Class 0 (Impostor): Denied all
    1: {"Unlock System", "Report Status", "Terminate Session"},       # Member 1 (Admin)
    2: {"Unlock System", "Report Status"},                            # Member 2 (Operator)
    3: {"Report Status"},                                             # Member 3 (Auditor)
    4: {"Report Status"},                                             # Member 4 (Auditor)
}


def check_rbac_permission(identity: int, command: str) -> bool:
    """Return True if `identity` is authorized to run `command`, else False."""
    if int(identity) <= IMPOSTOR_CLASS:
        return False
    allowed_commands = RBAC_PERMISSIONS.get(int(identity), {"Report Status"})
    return str(command).strip() in allowed_commands


def _unwrap_model(model_or_dict: Any) -> BaseBiometricModel:
    """Unwrap dict-saved artifacts (e.g. Member 1's best_face_model.joblib) into a BaseBiometricModel."""
    if isinstance(model_or_dict, dict) and "model" in model_or_dict:
        return model_or_dict["model"]
    return model_or_dict


class HierarchicalVerificationEngine:
    """3-Stage Hierarchical Multimodal Biometric Verification Engine."""

    def __init__(
        self,
        face_model: Any,
        voice_model: Any,
        fusion_model: Any,
        command_model: Optional[Any] = None,
        fusion_extractor: Optional[EarlyFeatureFusionExtractor] = None,
    ) -> None:
        self.face_model = _unwrap_model(face_model)
        self.voice_model = _unwrap_model(voice_model)
        self.fusion_model = _unwrap_model(fusion_model)
        self.command_model = _unwrap_model(command_model) if command_model is not None else None
        self.fusion_extractor = fusion_extractor

    @classmethod
    def from_saved_models(
        cls,
        face_path: Union[str, Path] = "models/face/best_face_model.joblib",
        voice_path: Union[str, Path] = "models/voice/best_voice_model.joblib",
        fusion_path: Union[str, Path] = "models/fusion/best_fusion_model.joblib",
        command_path: Union[str, Path] = "models/fingerprint/best_command_model.joblib",
    ) -> "HierarchicalVerificationEngine":
        """Load the trained models from Members 1, 2, 3, and 4 into the 3-stage engine."""
        import src.face_classifiers  # noqa: F401
        cmd_m = joblib.load(command_path) if Path(command_path).exists() else None
        return cls(
            face_model=joblib.load(face_path),
            voice_model=joblib.load(voice_path),
            fusion_model=joblib.load(fusion_path),
            command_model=cmd_m,
        )

    @staticmethod
    def _get_prediction(model: BaseBiometricModel, x_vec: Any) -> Tuple[int, float, int, float]:
        """
        Run `model.predict_proba(x_vec)` and return:
        (top_class, top_prob, best_genuine_class, best_genuine_prob)
        """
        inp = x_vec if isinstance(x_vec, tuple) else np.atleast_2d(x_vec)
        probs = np.asarray(model.predict_proba(inp), dtype=np.float64)[0]
        classes = np.asarray(model.classes_, dtype=np.int64)

        # Overall top predicted class (could be Class 0 Impostor or a Genuine ID)
        top_idx = int(np.argmax(probs))
        top_class = int(classes[top_idx])
        top_prob = float(probs[top_idx])

        # Highest probability among genuine enrolled subjects (excluding Class 0)
        genuine_mask = classes != IMPOSTOR_CLASS
        gen_classes = classes[genuine_mask]
        gen_probs = probs[genuine_mask]
        best_gen_idx = int(np.argmax(gen_probs))
        best_gen_class = int(gen_classes[best_gen_idx])
        best_gen_prob = float(gen_probs[best_gen_idx])

        return top_class, top_prob, best_gen_class, best_gen_prob

    def verify(
        self,
        face_vec: np.ndarray,
        voice_vec: Optional[np.ndarray] = None,
        print_vec: Optional[np.ndarray] = None,
        command: Optional[str] = None,
    ) -> DecisionRecord:
        """Execute Stage 1 -> Stage 2 -> Stage 3 hierarchical verification."""

        # =====================================================================
        # STAGE 1: FACE SCREENING
        # If P_face >= 0.88 -> VERIFIED at Stage 1
        # If P_face < 0.35  -> REJECTED_IMPOSTOR at Stage 1
        # Else (0.35 <= P_face < 0.88) -> Escalate to Stage 2
        # =====================================================================
        face_top_cls, _, face_id, p_face = self._get_prediction(self.face_model, face_vec)

        if face_top_cls != IMPOSTOR_CLASS and p_face >= 0.88:
            return DecisionRecord(
                status=STATUS_VERIFIED,
                identity=face_id,
                command=command or "N/A",
                stage=1,
                confidence=p_face,
            )

        if p_face < 0.35:
            return DecisionRecord(
                status=STATUS_REJECTED_IMPOSTOR,
                identity=IMPOSTOR_CLASS,
                command=command or "N/A",
                stage=1,
                confidence=p_face,
            )

        # =====================================================================
        # STAGE 2: VOICE & COMMAND + RBAC
        # If Voice_ID == Face_ID and P_voice >= 0.75 -> Check RBAC permission
        # Else -> Escalate to Stage 3
        # =====================================================================
        if voice_vec is None:
            raise ValueError("Stage 1 escalated to Stage 2, but voice_vec was not provided.")

        voice_id, p_voice, _, _ = self._get_prediction(self.voice_model, voice_vec)

        if command is not None:
            spoken_cmd = str(command)
        elif self.command_model is not None:
            spoken_cmd = str(self.command_model.predict(np.atleast_2d(voice_vec))[0])
        else:
            spoken_cmd = "Report Status"

        if voice_id == face_id and voice_id != IMPOSTOR_CLASS and p_voice >= 0.75:
            is_allowed = check_rbac_permission(voice_id, spoken_cmd)
            status = STATUS_VERIFIED if is_allowed else STATUS_PERMISSION_DENIED
            return DecisionRecord(
                status=status,
                identity=voice_id,
                command=spoken_cmd,
                stage=2,
                confidence=p_voice,
            )

        # =====================================================================
        # STAGE 3: FULL MULTIMODAL ARBITRATION (686D Early Fusion + SVM)
        # If P_fused >= 0.60 and class != 0 -> VERIFIED at Stage 3
        # Else -> REJECTED_IMPOSTOR at Stage 3
        # =====================================================================
        if print_vec is None:
            raise ValueError("Stage 2 escalated to Stage 3, but print_vec was not provided.")

        if isinstance(self.fusion_model, EarlyFusionMultimodalModel):
            fused_id, p_fused, _, _ = self._get_prediction(
                self.fusion_model, (face_vec, voice_vec, print_vec)
            )
        else:
            x_fused = (
                self.fusion_extractor.transform((face_vec, voice_vec, print_vec))
                if self.fusion_extractor is not None
                else np.hstack([np.atleast_2d(face_vec), np.atleast_2d(voice_vec), np.atleast_2d(print_vec)])
            )
            fused_id, p_fused, _, _ = self._get_prediction(self.fusion_model, x_fused)

        if fused_id != IMPOSTOR_CLASS and p_fused >= 0.60:
            return DecisionRecord(
                status=STATUS_VERIFIED,
                identity=fused_id,
                command=spoken_cmd,
                stage=3,
                confidence=p_fused,
            )

        return DecisionRecord(
            status=STATUS_REJECTED_IMPOSTOR,
            identity=IMPOSTOR_CLASS,
            command=spoken_cmd,
            stage=3,
            confidence=p_fused,
        )

    def verify_batch(
        self,
        X_face: Sequence[np.ndarray],
        X_voice: Sequence[np.ndarray],
        X_print: Sequence[np.ndarray],
        command: str = "Report Status",
    ) -> List[DecisionRecord]:
        """Verify a batch of samples through the 3-stage pipeline."""
        return [
            self.verify(X_face[i], X_voice[i], X_print[i], command=command)
            for i in range(len(X_face))
        ]
