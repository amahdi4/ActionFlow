from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from scripts.cli import main


class CliTests(unittest.TestCase):
    def test_launch_defaults_to_demo_profile(self) -> None:
        config = SimpleNamespace(runtime=SimpleNamespace(device="auto", log_level="INFO"))

        with patch("scripts.cli.load_default_config", return_value=config) as load_default_config:
            with patch("scripts.cli.launch_demo") as launch_demo:
                with patch("scripts.cli.configure_logging"):
                    exit_code = main(["launch"])

        self.assertEqual(exit_code, 0)
        load_default_config.assert_called_once_with("demo")
        launch_demo.assert_called_once_with(config)

    def test_dry_run_defaults_to_train_profile(self) -> None:
        config = SimpleNamespace(runtime=SimpleNamespace(device="auto", log_level="INFO"))
        pipeline = SimpleNamespace(dry_run_summary=lambda: {"ok": True})

        with patch("scripts.cli.load_default_config", return_value=config) as load_default_config:
            with patch("scripts.cli.TrainingPipeline", return_value=pipeline):
                with patch("scripts.cli.configure_logging"):
                    stream = io.StringIO()
                    with redirect_stdout(stream):
                        exit_code = main(["dry-run"])

        self.assertEqual(exit_code, 0)
        load_default_config.assert_called_once_with("train")
        self.assertIn('"ok": true', stream.getvalue())

    def test_device_override_is_applied_before_check_env(self) -> None:
        config = SimpleNamespace(runtime=SimpleNamespace(device="auto", log_level="INFO"))

        with patch("scripts.cli.load_default_config", return_value=config):
            with patch("scripts.cli.configure_logging"):
                with patch(
                    "scripts.cli.detect_device",
                    return_value=SimpleNamespace(to_dict=lambda: {"selected": "cpu"}),
                ) as detect_device:
                    stream = io.StringIO()
                    with redirect_stdout(stream):
                        exit_code = main(["--device", "cpu", "check-env"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(config.runtime.device, "cpu")
        detect_device.assert_called_once_with("cpu")
        self.assertIn('"selected": "cpu"', stream.getvalue())


if __name__ == "__main__":
    unittest.main()
