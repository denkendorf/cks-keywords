from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cks_keywords import FrozenCKS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("stage10_regression_report.json"))
    args = parser.parse_args()
    feature_path = args.project_root / "08_canonical_recoverability_refresh_and_development_method_tuning_scope_corrected" / "08_shared_candidate_features_fitted_on_development.csv.gz"
    expected_path = args.project_root / "10_frozen_test_evaluation_scope_corrected" / "10_test_selected_cks_rankings_top50.csv.gz"
    frame = pd.read_csv(feature_path)
    frame = frame.loc[frame["evaluation_role"].eq("test")].copy()
    profile = FrozenCKS.from_profile("paper1-w050-s40")
    observed = profile.score_feature_frame(frame, top_n=50)
    expected = pd.read_csv(expected_path)
    keys = ["record_id", "candidate_canonical", "candidate_rank"]
    joined = expected.merge(observed, on=keys, how="outer", suffixes=("_expected", "_observed"), indicator=True)
    score_difference = np.abs(joined["candidate_score_expected"] - joined["candidate_score_observed"])
    report = {
        "expected_rows": int(len(expected)),
        "observed_rows": int(len(observed)),
        "key_sets_identical": bool(joined["_merge"].eq("both").all()),
        "maximum_absolute_score_difference": float(score_difference.max(skipna=True)),
        "passed": bool(joined["_merge"].eq("both").all() and score_difference.fillna(np.inf).max() <= 1e-12),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
