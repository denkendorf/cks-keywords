from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

COMPONENT_COLUMNS = {
    "tfidf": "tfidf_feature",
    "df": "df_feature",
    "dispersion": "dispersion_feature",
    "domain_focus": "domain_focus_feature",
    "phrase_quality": "phrase_quality_feature",
}


def score_feature_frame(
    frame: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    output_column: str = "candidate_score",
) -> pd.DataFrame:
    missing = [column for column in COMPONENT_COLUMNS.values() if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing CKS feature columns: {missing}")
    unknown = set(weights) - set(COMPONENT_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown CKS components: {sorted(unknown)}")
    if abs(sum(float(v) for v in weights.values()) - 1.0) > 1e-9:
        raise ValueError("CKS weights must sum to 1.0.")
    result = frame.copy()
    result[output_column] = 0.0
    for component, weight in weights.items():
        result[output_column] += float(weight) * pd.to_numeric(
            result[COMPONENT_COLUMNS[component]], errors="raise"
        )
    if not result[output_column].between(-1e-9, 1.0 + 1e-9).all():
        raise ValueError("CKS scores fall outside [0,1].")
    return result


def rank_feature_frame(
    frame: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    minimum_score: float = 0.0,
    top_n: int = 10,
    record_id_column: str = "record_id",
    canonical_column: str = "candidate_canonical",
    exact_column: str = "candidate_exact",
    display_column: str = "candidate_display",
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    scored = score_feature_frame(frame, weights)
    required = [record_id_column, canonical_column, exact_column]
    missing = [c for c in required if c not in scored.columns]
    if missing:
        raise KeyError(f"Missing ranking columns: {missing}")
    if display_column not in scored.columns:
        scored[display_column] = scored[exact_column]
    ranked = scored.loc[scored["candidate_score"] + tolerance >= float(minimum_score)].copy()
    sort_columns = [record_id_column, "candidate_score"]
    ascending = [True, False]
    for component in ["domain_focus", "tfidf", "phrase_quality", "dispersion", "df"]:
        column = COMPONENT_COLUMNS[component]
        sort_columns.append(column)
        ascending.append(False)
    sort_columns.extend([canonical_column, exact_column])
    ascending.extend([True, True])
    ranked = ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    ranked = ranked.drop_duplicates([record_id_column, canonical_column], keep="first")
    ranked["candidate_rank"] = ranked.groupby(record_id_column, sort=False).cumcount() + 1
    ranked = ranked.loc[ranked["candidate_rank"] <= int(top_n)].copy()
    return ranked.reset_index(drop=True)
