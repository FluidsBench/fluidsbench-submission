from __future__ import annotations

import unittest

from scripts.add_prototype_methodologies import is_prototype_fixture


class PrototypeMethodologyMaintenanceTests(unittest.TestCase):
    def test_only_dummy_prototype_evidence_requires_generated_methodology(self) -> None:
        submission = {"approval": {"status": "prototype"}}

        self.assertTrue(
            is_prototype_fixture(submission, {"status": "prototype_dummy_data"})
        )
        self.assertFalse(
            is_prototype_fixture(submission, {"status": "pre_release_reference"})
        )
        self.assertFalse(
            is_prototype_fixture(submission, {"status": "submitted_evaluation"})
        )
        self.assertFalse(
            is_prototype_fixture({}, {"status": "prototype_dummy_data"})
        )


if __name__ == "__main__":
    unittest.main()
