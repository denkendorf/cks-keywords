from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_SPACE = re.compile(r"\s+")
_DASH = re.compile(r"[‐‑‒–—−-]+")
_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")


def normalize_keyword(value: object) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.lower().replace("’", "'")
    text = _DASH.sub(" ", text)
    text = _EDGE_PUNCT.sub("", text)
    text = _SPACE.sub(" ", text).strip()
    return text


@dataclass
class CanonicalMapper:
    mapping: dict[str, str] = field(default_factory=dict)

    def __call__(self, value: object) -> str:
        key = normalize_keyword(value)
        return self.mapping.get(key, key)

    def map_many(self, values: list[object]) -> list[str]:
        return [self(v) for v in values]

    @classmethod
    def from_csv(cls, path: str | Path) -> "CanonicalMapper":
        frame = pd.read_csv(path)
        target_candidates = ["canonical_label", "canonical_keyword", "canonical_key"]
        source_candidates = ["variant_key", "canonical_key", "mapping_base_key", "variant", "keyword"]
        target_cols = [c for c in target_candidates if c in frame.columns]
        source_cols = [c for c in source_candidates if c in frame.columns]
        if not target_cols or not source_cols:
            raise ValueError(f"Unsupported canonical mapping schema: {frame.columns.tolist()}")
        mapping: dict[str, str] = {}
        for _, row in frame.iterrows():
            target = next((normalize_keyword(row[c]) for c in target_cols if normalize_keyword(row[c])), "")
            if not target:
                continue
            mapping[target] = target
            for column in source_cols:
                source = normalize_keyword(row[column])
                if source:
                    prior = mapping.get(source)
                    if prior is not None and prior != target:
                        raise ValueError(f"Canonical mapping conflict for {source!r}: {prior!r} vs {target!r}")
                    mapping[source] = target
        return cls(mapping)
