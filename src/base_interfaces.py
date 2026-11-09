"""
src/base_interfaces.py
Zero-blocking base interface contracts for all biometric modalities (re-exported from interfaces.py).
"""
from interfaces import (  # noqa: F401
    FACE_DIM,
    FUSED_DIM,
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
