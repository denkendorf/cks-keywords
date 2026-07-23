from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CKSWeights:
    tfidf: float = 0.35
    df: float = 0.20
    dispersion: float = 0.20
    domain_focus: float = 0.15
    phrase_quality: float = 0.10

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(v < 0 for v in values.values()):
            raise ValueError("CKS weights must be non-negative.")
        if abs(sum(values.values()) - 1.0) > 1e-9:
            raise ValueError("CKS weights must sum to 1.0.")

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CKSWeights":
        return cls(**{k: float(values[k]) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CKSConfig:
    weights: CKSWeights = field(default_factory=CKSWeights)
    minimum_score: float = 0.0
    top_n: int = 10
    max_ngram: int = 4
    spacy_model: str = "en_core_web_sm"
    profile_name: str | None = None
    configuration_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be in [0, 1].")
        if self.top_n < 1:
            raise ValueError("top_n must be positive.")
        if self.max_ngram < 1:
            raise ValueError("max_ngram must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "minimum_score": float(self.minimum_score),
            "top_n": int(self.top_n),
            "max_ngram": int(self.max_ngram),
            "spacy_model": self.spacy_model,
            "profile_name": self.profile_name,
            "configuration_id": self.configuration_id,
        }
