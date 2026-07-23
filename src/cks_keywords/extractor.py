from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .canonical import CanonicalMapper
from .candidates import CandidateGenerator
from .config import CKSConfig
from .features import FittedState, document_feature_frame, fit_state
from .scoring import rank_feature_frame


class CKSKeywordExtractor:
    def __init__(
        self,
        config: CKSConfig,
        generator: CandidateGenerator,
        mapper: CanonicalMapper | None = None,
        state: FittedState | None = None,
    ) -> None:
        self.config = config
        self.generator = generator
        self.mapper = mapper or CanonicalMapper()
        self.state = state

    def fit(
        self,
        documents: pd.DataFrame,
        *,
        text_column: str = "abstract",
        id_column: str = "record_id",
        keyword_column: str | None = None,
        keyword_separator: str = ";",
    ) -> "CKSKeywordExtractor":
        required = [id_column, text_column]
        missing = [c for c in required if c not in documents.columns]
        if missing:
            raise KeyError(f"Missing fit columns: {missing}")
        texts = dict(zip(documents[id_column].astype(str), documents[text_column].fillna("").astype(str)))
        author_keywords = None
        if keyword_column:
            author_keywords = {
                str(row[id_column]): [k.strip() for k in str(row[keyword_column]).split(keyword_separator) if k.strip()]
                for _, row in documents.iterrows()
            }
        self.state = fit_state(texts, self.generator, self.mapper, author_keywords)
        return self

    def _require_fitted(self) -> FittedState:
        if self.state is None:
            raise RuntimeError("CKSKeywordExtractor is not fitted. Call fit() or load a frozen profile.")
        return self.state

    def extract_keywords(
        self,
        abstract: str,
        *,
        title: str = "",
        record_id: str = "document",
        top_n: int | None = None,
        return_features: bool = True,
    ):
        state = self._require_fitted()
        frame = document_feature_frame(record_id, abstract, self.generator, state, title=title)
        if frame.empty:
            return [] if not return_features else frame
        ranked = rank_feature_frame(
            frame,
            self.config.weights.to_dict(),
            minimum_score=self.config.minimum_score,
            top_n=top_n or self.config.top_n,
        )
        if return_features:
            return ranked
        return [
            (row.candidate_display, float(row.candidate_score))
            for row in ranked.itertuples(index=False)
        ]

    def extract_keywords_batch(
        self,
        documents: pd.DataFrame,
        *,
        text_column: str = "abstract",
        title_column: str | None = "title",
        id_column: str = "record_id",
        top_n: int | None = None,
    ) -> pd.DataFrame:
        outputs = []
        for _, row in documents.iterrows():
            outputs.append(
                self.extract_keywords(
                    str(row.get(text_column, "")),
                    title=str(row.get(title_column, "")) if title_column else "",
                    record_id=str(row[id_column]),
                    top_n=top_n,
                    return_features=True,
                )
            )
        nonempty = [frame for frame in outputs if not frame.empty]
        return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()

    def save(self, directory: str | Path) -> None:
        state = self._require_fitted()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")
        (target / "candidate_rule.json").write_text(json.dumps(self.generator.rule, indent=2), encoding="utf-8")
        pd.DataFrame(
            [{"variant_key": k, "canonical_label": v} for k, v in sorted(self.mapper.mapping.items())]
        ).to_csv(target / "canonical_mapping.csv", index=False)
        state.save(target)
