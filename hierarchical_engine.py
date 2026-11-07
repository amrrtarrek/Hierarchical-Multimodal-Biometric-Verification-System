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

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from early_fusion import EarlyFeatureFusionExtractor, EarlyFusionMultimodalModel
from interfaces import (
    FACE_DIM,
    IMPOSTOR_CLASS,
    PRINT_DIM,
    STATUS_PERMISSION_DENIED,
    STATUS_REJECTED_IMPOSTOR,
    STATUS_VERIFIED,
    VOICE_DIM,
    BaseBiometricModel,
    BaseFeatureExtractor,
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
    """Evaluate RBAC authorization for a verified identity and spoken command."""
    if int(identity) <= IMPOSTOR_CLASS:
        return False
    allowed = RBAC_PERMISSIONS.get(int(identity), {"Report Status"})
    return str(command).strip() in allowed


class HierarchicalVerificationEngine:
    """3-Stage Hierarchical Multimodal Biometric Verification Engine."""

    def __init__(
        self,
        face_model: BaseBiometricModel,
        voice_model: BaseBiometricModel,
        fusion_model: BaseBiometricModel,
        command_model: Optional[BaseBiometricModel] = None,
        rbac_checker: Optional[Any] = None,
        face_extractor: Optional[BaseFeatureExtractor] = None,
        voice_extractor: Optional[BaseFeatureExtractor] = None,
        print_extractor: Optional[BaseFeatureExtractor] = None,
        fusion_extractor: Optional[EarlyFeatureFusionExtractor] = None,
    ) -> None:
        self.face_model = face_model
        self.voice_model = voice_model
        self.fusion_model = fusion_model
        self.command_model = command_model
        self.rbac_checker = rbac_checker
        self.face_extractor = face_extractor
        self.voice_extractor = voice_extractor
        self.print_extractor = print_extractor
        self.fusion_extractor = fusion_extractor

    @staticmethod
    def _predict_probs(model: BaseBiometricModel, x: np.ndarray) -> Tuple[int, float, int, float]:
        """Return (top_class, top_prob, best_genuine_class, best_genuine_prob)."""
        probs = np.asarray(model.predict_proba(x), dtype=np.float64).ravel()
        classes = getattr(model, "classes_", None)
        classes = np.asarray(classes, dtype=np.int64) if classes is not None else np.arange(len(probs))

        top_idx = int(np.argmax(probs))
        top_cls, top_prob = int(classes[top_idx]), float(probs[top_idx])

        gen_mask = classes != IMPOSTOR_CLASS
        gen_indices = np.where(gen_mask)[0]
        best_gen_idx = int(gen_indices[int(np.argmax(probs[gen_indices]))])
        return top_cls, top_prob, int(classes[best_gen_idx]), float(probs[best_gen_idx])

    @staticmethod
    def _to_vector(sample: Any, dim: int, extractor: Optional[BaseFeatureExtractor]) -> np.ndarray:
        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim == 1 and arr.size == dim:
            return arr.reshape(1, dim)
        if arr.ndim == 2 and arr.shape == (1, dim):
            return arr
        if extractor is not None:
            return extractor.transform(sample)
        raise ValueError(f"Expected {dim}D feature vector or a configured extractor.")

    def _is_authorized(self, identity: int, command: str) -> bool:
        if self.rbac_checker is not None:
            if hasattr(self.rbac_checker, "is_authorized"):
                return bool(self.rbac_checker.is_authorized(identity, command))
            if callable(self.rbac_checker):
                return bool(self.rbac_checker(identity, command))
        return check_rbac_permission(identity, command)

    def verify(
        self,
        face_input: Any,
        voice_input: Optional[Any] = None,
        print_input: Optional[Any] = None,
        command: Optional[str] = None,
    ) -> DecisionRecord:
        """Execute Stage 1 -> Stage 2 -> Stage 3 hierarchical verification."""
        # --- STAGE 1: FACE SCREENING ---
        x_face = self._to_vector(face_input, FACE_DIM, self.face_extractor)
        f_top_cls, _, f_gen_cls, p_face = self._predict_probs(self.face_model, x_face)

        if f_top_cls != IMPOSTOR_CLASS and p_face >= 0.88:
            return DecisionRecord(
                status=STATUS_VERIFIED,
                identity=f_gen_cls,
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

        # --- STAGE 2: VOICE & COMMAND + RBAC ---
        if voice_input is None:
            raise ValueError("Escalated to Stage 2, but voice_input was not provided.")
        x_voice = self._to_vector(voice_input, VOICE_DIM, self.voice_extractor)
        v_top_cls, v_top_prob, _, p_voice_gen = self._predict_probs(self.voice_model, x_voice)
        p_voice = v_top_prob if v_top_cls != IMPOSTOR_CLASS else p_voice_gen

        if command is not None:
            spoken_cmd = str(command)
        elif self.command_model is not None:
            spoken_cmd = str(np.asarray(self.command_model.predict(x_voice)).ravel()[0])
        else:
            spoken_cmd = "Report Status"

        if v_top_cls == f_gen_cls and v_top_cls != IMPOSTOR_CLASS and p_voice >= 0.75:
            status = STATUS_VERIFIED if self._is_authorized(v_top_cls, spoken_cmd) else STATUS_PERMISSION_DENIED
            return DecisionRecord(
                status=status,
                identity=v_top_cls,
                command=spoken_cmd,
                stage=2,
                confidence=p_voice,
            )

        # --- STAGE 3: FULL MULTIMODAL ARBITRATION ---
        if print_input is None:
            raise ValueError("Escalated to Stage 3, but print_input was not provided.")
        x_print = self._to_vector(print_input, PRINT_DIM, self.print_extractor)

        if isinstance(self.fusion_model, EarlyFusionMultimodalModel):
            probs = np.asarray(self.fusion_model.predict_proba((x_face, x_voice, x_print)), dtype=np.float64).ravel()
            classes = np.asarray(self.fusion_model.classes_, dtype=np.int64)
            top_idx = int(np.argmax(probs))
            m_top_cls, p_fused = int(classes[top_idx]), float(probs[top_idx])
        else:
            x_fused = (
                self.fusion_extractor.transform((x_face, x_voice, x_print))
                if self.fusion_extractor is not None
                else np.hstack([x_face, x_voice, x_print]).astype(np.float32)
            )
            m_top_cls, p_fused, _, _ = self._predict_probs(self.fusion_model, x_fused)

        if m_top_cls != IMPOSTOR_CLASS and p_fused >= 0.60:
            return DecisionRecord(
                status=STATUS_VERIFIED,
                identity=m_top_cls,
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
        X_face: Sequence[Any],
        X_voice: Sequence[Any],
        X_print: Sequence[Any],
        command: str = "Report Status",
    ) -> List[DecisionRecord]:
        return [
            self.verify(X_face[i], X_voice[i], X_print[i], command=command)
            for i in range(len(X_face))
        ]
