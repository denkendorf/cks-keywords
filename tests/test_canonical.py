import tempfile
import unittest
from pathlib import Path

import pandas as pd
from cks_keywords.canonical import CanonicalMapper, normalize_keyword


class CanonicalTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_keyword("  Sign–Language  "), "sign language")

    def test_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.csv"
            pd.DataFrame(
                {"variant_key": ["asl"], "canonical_key": ["american sign language"], "canonical_label": ["american sign language"]}
            ).to_csv(path, index=False)
            mapper = CanonicalMapper.from_csv(path)
            self.assertEqual(mapper("ASL"), "american sign language")
            self.assertEqual(mapper("handshape"), "handshape")


if __name__ == "__main__":
    unittest.main()
