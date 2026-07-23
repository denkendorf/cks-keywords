import unittest

import pandas as pd
from cks_keywords.scoring import rank_feature_frame, score_feature_frame


class ScoringTests(unittest.TestCase):
    def test_convex_score_and_ranking(self):
        frame = pd.DataFrame(
            {
                "record_id": ["d1", "d1", "d1"],
                "candidate_display": ["alpha", "beta", "alpha variant"],
                "candidate_exact": ["alpha", "beta", "alpha variant"],
                "candidate_canonical": ["alpha", "beta", "alpha"],
                "tfidf_feature": [1.0, 0.0, 0.8],
                "df_feature": [0.0, 1.0, 0.0],
                "dispersion_feature": [0.0, 0.0, 0.0],
                "domain_focus_feature": [0.0, 0.0, 0.0],
                "phrase_quality_feature": [0.0, 0.0, 0.0],
            }
        )
        weights = {"tfidf": 0.5, "df": 0.5, "dispersion": 0.0, "domain_focus": 0.0, "phrase_quality": 0.0}
        scored = score_feature_frame(frame, weights)
        self.assertTrue(scored["candidate_score"].between(0, 1).all())
        ranked = rank_feature_frame(frame, weights, top_n=10)
        self.assertEqual(ranked["candidate_canonical"].nunique(), 2)
        self.assertEqual(ranked["candidate_rank"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
