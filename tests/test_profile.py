import unittest

from cks_keywords import FrozenCKS


class ProfileTests(unittest.TestCase):
    def test_frozen_profile_manifest(self):
        profile = FrozenCKS.from_profile("paper1-w050-s40")
        self.assertTrue(profile.verify_manifest())
        self.assertEqual(profile.manifest["configuration_id"], "W050_S40")
        self.assertGreater(profile.manifest["term_statistics_rows"], 70000)


if __name__ == "__main__":
    unittest.main()
