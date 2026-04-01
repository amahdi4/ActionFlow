from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils.device import detect_device


def build_fake_torch(*, cuda: bool = False, mps: bool = False, version: str = "2.0.0"):
    return SimpleNamespace(
        __version__=version,
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


class DeviceDetectionTests(unittest.TestCase):
    def test_auto_prefers_mps_on_macos(self) -> None:
        fake_torch = build_fake_torch(cuda=False, mps=True, version="2.1.0")

        with patch("utils.device._import_torch", return_value=fake_torch):
            with patch("utils.device.platform.system", return_value="Darwin"):
                with patch("utils.device.platform.machine", return_value="arm64"):
                    device = detect_device("auto")

        self.assertEqual(device.selected, "mps")
        self.assertEqual(device.available_devices, ["mps", "cpu"])
        self.assertEqual(device.torch_version, "2.1.0")

    def test_auto_prefers_cuda_off_macos(self) -> None:
        fake_torch = build_fake_torch(cuda=True, mps=False)

        with patch("utils.device._import_torch", return_value=fake_torch):
            with patch("utils.device.platform.system", return_value="Linux"):
                with patch("utils.device.platform.machine", return_value="x86_64"):
                    device = detect_device("auto")

        self.assertEqual(device.selected, "cuda")
        self.assertEqual(device.available_devices, ["cuda", "cpu"])
        self.assertTrue(device.mixed_precision)

    def test_requested_cuda_falls_back_to_mps_when_available(self) -> None:
        fake_torch = build_fake_torch(cuda=False, mps=True)

        with patch("utils.device._import_torch", return_value=fake_torch):
            with patch("utils.device.platform.system", return_value="Darwin"):
                with patch("utils.device.platform.machine", return_value="arm64"):
                    device = detect_device("cuda")

        self.assertEqual(device.selected, "mps")
        self.assertIn("using `mps` instead", device.reason)

    def test_missing_torch_falls_back_to_cpu(self) -> None:
        with patch("utils.device._import_torch", return_value=None):
            with patch("utils.device.platform.system", return_value="Darwin"):
                with patch("utils.device.platform.machine", return_value="arm64"):
                    device = detect_device("auto")

        self.assertEqual(device.selected, "cpu")
        self.assertEqual(device.available_devices, ["cpu"])
        self.assertIsNone(device.torch_version)

    def test_invalid_preference_raises(self) -> None:
        with self.assertRaises(ValueError):
            detect_device("tpu")


if __name__ == "__main__":
    unittest.main()
