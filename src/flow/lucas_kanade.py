from __future__ import annotations

from typing import Any

import numpy as np

from flow.base import FlowEstimator, FlowField


class LucasKanadeEstimator(FlowEstimator):
    name = "lucas_kanade"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    @classmethod
    def availability_details(cls) -> str:
        return "Requires OpenCV (`opencv-python`). Produces sparse tracks that will later be rasterized into heatmaps."

    def estimate(self, frame_a: Any, frame_b: Any) -> FlowField:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Lucas-Kanade tracking requires `opencv-python`."
            ) from exc

        gray_a = _to_grayscale(frame_a, cv2)
        gray_b = _to_grayscale(frame_b, cv2)
        corners = cv2.goodFeaturesToTrack(
            gray_a,
            maxCorners=int(self.params.get("max_corners", 200)),
            qualityLevel=float(self.params.get("quality_level", 0.3)),
            minDistance=float(self.params.get("min_distance", 7.0)),
            blockSize=int(self.params.get("block_size", 7)),
        )
        horizontal = np.zeros_like(gray_a, dtype=np.float32)
        vertical = np.zeros_like(gray_a, dtype=np.float32)
        magnitude = np.zeros_like(gray_a, dtype=np.float32)
        if corners is None:
            return FlowField(
                horizontal=horizontal,
                vertical=vertical,
                magnitude=magnitude,
                metadata={"num_tracks": 0, "tracks": []},
            )

        next_corners, status, _ = cv2.calcOpticalFlowPyrLK(
            gray_a,
            gray_b,
            corners,
            None,
            winSize=(
                int(self.params.get("win_size", 15)),
                int(self.params.get("win_size", 15)),
            ),
            maxLevel=int(self.params.get("max_level", 2)),
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(self.params.get("criteria_count", 10)),
                float(self.params.get("criteria_eps", 0.03)),
            ),
        )
        if next_corners is None or status is None:
            return FlowField(
                horizontal=horizontal,
                vertical=vertical,
                magnitude=magnitude,
                metadata={"num_tracks": 0, "tracks": []},
            )

        good_new = next_corners[status.reshape(-1) == 1]
        good_old = corners[status.reshape(-1) == 1]
        tracks: list[dict[str, float]] = []
        height, width = gray_a.shape
        for new_point, old_point in zip(good_new, good_old):
            new_x, new_y = new_point.ravel()
            old_x, old_y = old_point.ravel()
            dx = float(new_x - old_x)
            dy = float(new_y - old_y)
            px = min(max(int(round(new_x)), 0), width - 1)
            py = min(max(int(round(new_y)), 0), height - 1)
            horizontal[py, px] = dx
            vertical[py, px] = dy
            magnitude[py, px] = float((dx * dx + dy * dy) ** 0.5)
            tracks.append(
                {
                    "x": float(new_x),
                    "y": float(new_y),
                    "dx": dx,
                    "dy": dy,
                }
            )

        return FlowField(
            horizontal=horizontal,
            vertical=vertical,
            magnitude=magnitude,
            metadata={"num_tracks": len(tracks), "tracks": tracks[:50]},
        )


def _to_grayscale(frame: Any, cv2) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        grayscale = array
    elif array.ndim == 3 and array.shape[2] >= 3:
        grayscale = cv2.cvtColor(array[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Expected a grayscale or RGB frame for motion tracking.")
    return grayscale.astype(np.uint8, copy=False)
