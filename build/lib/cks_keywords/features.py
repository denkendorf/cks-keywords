from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .canonical import CanonicalMapper
from .candidates import Candidate, CandidateGenerator


@dataclass
class FittedState:
    n_documents: int
    term_statistics: pd.DataFrame
    length_prior: dict[int, float]
    mapper: CanonicalMapper

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.term_statistics.to_csv(target / "term_statistics.csv.gz", index=False, compression="gzip")
        (target / "state.json").write_text(
            json.dumps({"n_documents": self.n_documents, "length_prior": self.length_prior}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path, mapper: CanonicalMapper) -> "FittedState":
        target = Path(directory)
        meta = json.loads((target / "state.json").read_text(encoding="utf-8"))
        return cls(
            int(meta["n_documents"]),
            pd.read_csv(target / "term_statistics.csv.gz"),
            {int(k): float(v) for k, v in meta["length_prior"].items()},
            mapper,
        )


def fit_state(
    texts: Mapping[str, str],
    generator: CandidateGenerator,
    mapper: CanonicalMapper,
    author_keywords: Mapping[str, Iterable[str]] | None = None,
) -> FittedState:
    if not texts:
        raise ValueError("At least one fitting document is required.")
    rows: list[dict] = []
    for record_id, text in texts.items():
        for candidate in generator.generate(text):
            canonical = mapper(candidate.lemma or candidate.exact)
            rows.append(
                {
                    "record_id": str(record_id),
                    "candidate_canonical": canonical,
                    "count": int(candidate.count),
                }
            )
    occurrence = pd.DataFrame(rows)
    if occurrence.empty:
        raise ValueError("No admissible candidates were generated.")
    occurrence = occurrence.groupby(["record_id", "candidate_canonical"], as_index=False)["count"].sum()
    n_documents = len(texts)
    grouped = occurrence.groupby("candidate_canonical", sort=False)
    stats = grouped.agg(df_fit=("record_id", "nunique"), total_count=("count", "sum")).reset_index()
    dispersion = {}
    for candidate, group in grouped:
        counts = group["count"].to_numpy(dtype=float)
        probs = counts / counts.sum()
        entropy = -float(np.sum(probs * np.log(probs))) if len(probs) else 0.0
        dispersion[candidate] = entropy / math.log(n_documents) if n_documents > 1 else 0.0
    stats["dispersion_feature"] = stats["candidate_canonical"].map(dispersion).astype(float)
    stats["df_feature"] = np.log1p(stats["df_fit"]) / math.log1p(n_documents)
    stats["idf_fit"] = np.log((n_documents + 1) / (stats["df_fit"] + 1)) + 1.0

    gold_df: dict[str, int] = {}
    length_counts: dict[int, int] = {}
    if author_keywords:
        for _, keywords in author_keywords.items():
            canonical_set = {mapper(k) for k in keywords if mapper(k)}
            for keyword in canonical_set:
                gold_df[keyword] = gold_df.get(keyword, 0) + 1
                length = len(keyword.split())
                length_counts[length] = length_counts.get(length, 0) + 1
    gmax = max(gold_df.values(), default=0)
    stats["domain_gold_df_fit"] = stats["candidate_canonical"].map(gold_df).fillna(0).astype(int)
    if gmax:
        stats["domain_focus_feature"] = np.log1p(stats["domain_gold_df_fit"]) / math.log1p(gmax)
    else:
        stats["domain_focus_feature"] = 0.0
    maximum_length_count = max(length_counts.values(), default=1)
    length_prior = {length: count / maximum_length_count for length, count in length_counts.items()}
    for length in range(1, generator.rule.get("max_ngram", 4) + 1):
        length_prior.setdefault(length, 1.0)
    return FittedState(n_documents, stats, length_prior, mapper)


def document_feature_frame(
    record_id: str,
    text: str,
    generator: CandidateGenerator,
    state: FittedState,
    *,
    title: str = "",
) -> pd.DataFrame:
    candidates = generator.generate(text)
    if not candidates:
        return pd.DataFrame()
    stats = state.term_statistics.set_index("candidate_canonical", drop=False)
    rows: list[dict] = []
    length_totals: dict[int, int] = {}
    for candidate in candidates:
        length_totals[candidate.ngram_length] = length_totals.get(candidate.ngram_length, 0) + candidate.count
    for candidate in candidates:
        canonical = state.mapper(candidate.lemma or candidate.exact)
        known = stats.loc[canonical] if canonical in stats.index else None
        df_fit = int(known["df_fit"]) if known is not None else 0
        idf = float(known["idf_fit"]) if known is not None else math.log((state.n_documents + 1) / 1) + 1
        tf = candidate.count / max(1, length_totals[candidate.ngram_length])
        length_prior = float(state.length_prior.get(candidate.ngram_length, 0.0))
        rows.append(
            {
                "record_id": str(record_id),
                "candidate_display": candidate.display,
                "candidate_exact": candidate.exact,
                "candidate_lemma_norm": candidate.lemma,
                "candidate_canonical": canonical,
                "ngram_length": candidate.ngram_length,
                "count_in_abstract": candidate.count,
                "count_in_title": title.lower().count(candidate.exact) if title else 0,
                "tf": tf,
                "df_fit": df_fit,
                "df_feature": float(known["df_feature"]) if known is not None else 0.0,
                "idf_fit": idf,
                "tfidf_raw": tf * idf,
                "dispersion_feature": float(known["dispersion_feature"]) if known is not None else 0.0,
                "domain_gold_df_fit": int(known["domain_gold_df_fit"]) if known is not None else 0,
                "domain_focus_feature": float(known["domain_focus_feature"]) if known is not None else 0.0,
                "phrase_structural_quality": candidate.structural_quality,
                "phrase_length_prior": length_prior,
                "phrase_quality_feature": 0.7 * candidate.structural_quality + 0.3 * length_prior,
            }
        )
    frame = pd.DataFrame(rows)
    low = frame["tfidf_raw"].min()
    high = frame["tfidf_raw"].max()
    frame["tfidf_feature"] = 0.0 if high == low else (frame["tfidf_raw"] - low) / (high - low)
    return frame
