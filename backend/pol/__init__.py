"""
Proof of Learning (PoL) Module for Blyan Network
Secure evaluation system with anti-gaming measures
"""

from .proof_of_learning import (
    ProofOfLearningEvaluator,
    PoLSubmission,
    EvaluationMetrics,
    DatasetType,
    TestSample
)

from .anti_gaming import (
    AntiGamingSystem,
    AbusePattern
)

__all__ = [
    'ProofOfLearningEvaluator',
    'PoLSubmission',
    'EvaluationMetrics',
    'DatasetType',
    'TestSample',
    'AntiGamingSystem',
    'AbusePattern'
]