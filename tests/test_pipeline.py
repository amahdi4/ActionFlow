from __future__ import annotations

import unittest

from configs.loader import load_default_config
from training.pipeline import TrainingPipeline


class TrainingPipelineTests(unittest.TestCase):
    def test_dry_run_summary_contains_expected_sections(self) -> None:
        config = load_default_config("train")
        pipeline = TrainingPipeline(config)
        summary = pipeline.dry_run_summary()

        self.assertIn("project", summary)
        self.assertIn("runtime", summary)
        self.assertIn("dataset", summary)
        self.assertIn("flow", summary)
        self.assertIn("representation", summary)
        self.assertIn("model", summary)
        self.assertEqual(summary["flow"]["name"], "farneback")


if __name__ == "__main__":
    unittest.main()
