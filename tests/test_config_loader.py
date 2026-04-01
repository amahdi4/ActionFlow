from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from configs.loader import load_config, load_default_config


class ConfigLoaderTests(unittest.TestCase):
    def test_load_default_train_config(self) -> None:
        config = load_default_config("train")
        self.assertEqual(config.project.name, "ActionFlow")
        self.assertEqual(config.flow.backend, "farneback")
        self.assertEqual(config.model.num_classes, len(config.data.class_names))

    def test_load_json_config(self) -> None:
        payload = {
            "project": {"name": "TestProject"},
            "data": {"class_names": ["one", "two"]},
            "model": {"num_classes": 99, "input_channels": 99},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(config_path)

        self.assertEqual(config.project.name, "TestProject")
        self.assertEqual(config.model.num_classes, 2)
        self.assertEqual(config.model.input_channels, config.data.clip_length * 2)


if __name__ == "__main__":
    unittest.main()

