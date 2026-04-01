from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from demo.analysis import analyze_live_frame, reset_live_outputs
from flow.base import FlowField


class DummyEstimator:
    def estimate(self, frame_a, frame_b) -> FlowField:
        difference = np.abs(frame_b.astype(np.float32) - frame_a.astype(np.float32)).mean(axis=2) / 255.0
        return FlowField(
            horizontal=np.zeros_like(difference, dtype=np.float32),
            vertical=np.zeros_like(difference, dtype=np.float32),
            magnitude=difference.astype(np.float32),
        )


class AlternatingWaveEstimator:
    def __init__(self) -> None:
        self.step = 0

    def estimate(self, frame_a, frame_b) -> FlowField:
        self.step += 1
        horizontal = np.zeros((32, 32), dtype=np.float32)
        vertical = np.zeros((32, 32), dtype=np.float32)
        horizontal[:16, 10:24] = 3.0 if self.step % 2 == 1 else -3.0
        magnitude = np.abs(horizontal)
        return FlowField(horizontal=horizontal, vertical=vertical, magnitude=magnitude)


class DemoAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_summary = {"runtime": {"device": "cpu", "selected": "cpu"}}
        self.backend_summaries = {
            "farneback": {
                "available": True,
                "availability_details": "Dummy estimator for tests.",
            },
            "raft": {
                "available": False,
                "availability_details": "Unavailable in tests.",
            },
        }

    def test_reset_live_outputs_returns_idle_state(self) -> None:
        image, status, guidance, metrics, details, state = reset_live_outputs(
            "farneback",
            self.runtime_summary,
            self.backend_summaries,
        )

        self.assertIsNone(image)
        self.assertIn("Waiting for live camera frames", status)
        self.assertIn("What To Do", guidance)
        self.assertEqual(metrics["analysis_mode"], "idle")
        self.assertEqual(state["frame_index"], 0)
        payload = json.loads(details)
        self.assertEqual(payload["status"], "waiting_for_camera")
        copy.deepcopy(state)
        self.assertIn("session_id", state)
        self.assertNotIn("background_model", state)

    def test_live_analyzer_builds_baseline_then_detects_motion(self) -> None:
        _, _, _, _, _, state = reset_live_outputs(
            "farneback",
            self.runtime_summary,
            self.backend_summaries,
        )
        estimators = {"farneback": DummyEstimator()}
        first_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        second_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        second_frame[:, 12:20, :] = 255

        first_output = analyze_live_frame(
            first_frame,
            "farneback",
            state,
            estimators,
            self.backend_summaries,
            self.runtime_summary,
        )
        self.assertEqual(first_output[3]["motion_energy"], 0.0)
        self.assertEqual(first_output[5]["frame_index"], 1)

        second_output = analyze_live_frame(
            second_frame,
            "farneback",
            first_output[5],
            estimators,
            self.backend_summaries,
            self.runtime_summary,
        )
        self.assertIsNotNone(second_output[0])
        self.assertGreater(second_output[3]["motion_energy"], 0.0)
        self.assertIn("Action hypothesis", second_output[1])
        self.assertIn("action_hypothesis", second_output[3])
        self.assertEqual(second_output[5]["frame_index"], 2)

    def test_live_analyzer_falls_back_when_backend_unavailable(self) -> None:
        _, _, _, _, _, state = reset_live_outputs(
            "raft",
            self.runtime_summary,
            self.backend_summaries,
        )
        frame_a = np.zeros((24, 24, 3), dtype=np.uint8)
        frame_b = np.full((24, 24, 3), 120, dtype=np.uint8)

        first_output = analyze_live_frame(
            frame_a,
            "raft",
            state,
            {},
            self.backend_summaries,
            self.runtime_summary,
        )
        second_output = analyze_live_frame(
            frame_b,
            "raft",
            first_output[5],
            {},
            self.backend_summaries,
            self.runtime_summary,
        )

        self.assertEqual(second_output[3]["analysis_mode"], "frame_difference_fallback")
        self.assertGreater(second_output[3]["motion_energy"], 0.0)
        payload = json.loads(second_output[4])
        self.assertEqual(payload["selected_backend"], "raft")

    def test_live_analyzer_can_surface_wave_hypothesis(self) -> None:
        _, _, _, _, _, state = reset_live_outputs(
            "farneback",
            self.runtime_summary,
            self.backend_summaries,
        )
        estimators = {"farneback": AlternatingWaveEstimator()}
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        output = analyze_live_frame(
            frame,
            "farneback",
            state,
            estimators,
            self.backend_summaries,
            self.runtime_summary,
        )
        for _ in range(3):
            output = analyze_live_frame(
                frame,
                "farneback",
                output[5],
                estimators,
                self.backend_summaries,
                self.runtime_summary,
            )

        self.assertIn("Wave", output[1])
        self.assertEqual(output[3]["action_hypothesis"], "Wave")
        self.assertGreater(output[3]["action_confidence"], 0.3)

    def test_live_analyzer_reports_camera_shift_for_global_translation(self) -> None:
        _, _, _, _, _, state = reset_live_outputs(
            "farneback",
            self.runtime_summary,
            self.backend_summaries,
        )
        frame_a = np.zeros((48, 48, 3), dtype=np.uint8)
        frame_a[12:36, 10:22, :] = 255
        frame_b = np.zeros_like(frame_a)
        frame_b[:, 4:, :] = frame_a[:, :-4, :]

        first_output = analyze_live_frame(
            frame_a,
            "farneback",
            state,
            {},
            self.backend_summaries,
            self.runtime_summary,
        )
        second_output = analyze_live_frame(
            frame_b,
            "farneback",
            first_output[5],
            {},
            self.backend_summaries,
            self.runtime_summary,
        )

        self.assertGreater(second_output[3]["camera_shift_px"], 1.0)
        self.assertIn("shift_px", second_output[1])

    def test_live_overlay_preserves_input_resolution(self) -> None:
        _, _, _, _, _, state = reset_live_outputs(
            "farneback",
            self.runtime_summary,
            self.backend_summaries,
        )
        frame_a = np.zeros((360, 640, 3), dtype=np.uint8)
        frame_b = np.zeros_like(frame_a)
        frame_b[:, 220:360, :] = 255

        first_output = analyze_live_frame(
            frame_a,
            "farneback",
            state,
            {"farneback": DummyEstimator()},
            self.backend_summaries,
            self.runtime_summary,
        )
        second_output = analyze_live_frame(
            frame_b,
            "farneback",
            first_output[5],
            {"farneback": DummyEstimator()},
            self.backend_summaries,
            self.runtime_summary,
        )

        self.assertEqual(second_output[0].shape[:2], frame_b.shape[:2])


if __name__ == "__main__":
    unittest.main()
