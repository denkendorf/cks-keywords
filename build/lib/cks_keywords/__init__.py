"""Composite Keyword Scoring (CKS) for metadata-aware keyword reconstruction."""

from .config import CKSConfig, CKSWeights
from .extractor import CKSKeywordExtractor
from .profiles import FrozenCKS, available_profiles
from .scoring import rank_feature_frame, score_feature_frame

__all__ = [
    "CKSConfig",
    "CKSWeights",
    "CKSKeywordExtractor",
    "FrozenCKS",
    "available_profiles",
    "score_feature_frame",
    "rank_feature_frame",
]
__version__ = "0.1.1"
