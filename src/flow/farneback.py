from __future__ import annotations

from typing import Any

import numpy as np

from flow.base import FlowEstimator, FlowField


class FarnebackEstimator(FlowEstimator):
    name = "farneback"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    @classmethod
    def availability_details(cls) -> str:
        return "Requires OpenCV (`opencv-python`)."

    def estimate(self, frame_a: Any, frame_b: Any) -> FlowField:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Farneback optical flow requires `opencv-python`."
            ) from exc

        gray_a = _to_grayscale(frame_a, cv2)
        gray_b = _to_grayscale(frame_b, cv2)
        flow = cv2.calcOpticalFlowFarneback(
            gray_a,
            gray_b,
            None,
            pyr_scale=float(self.params.get("pyr_scale", 0.5)),
            levels=int(self.params.get("levels", 3)),
            winsize=int(self.params.get("winsize", 15)),
            iterations=int(self.params.get("iterations", 3)),
            poly_n=int(self.params.get("poly_n", 5)),
            poly_sigma=float(self.params.get("poly_sigma", 1.2)),
            flags=int(self.params.get("flags", 0)),
        )
        horizontal = flow[..., 0].astype(np.float32)
        vertical = flow[..., 1].astype(np.float32)
        magnitude, angle = cv2.cartToPolar(horizontal, vertical)
        return FlowField(
            horizontal=horizontal,
            vertical=vertical,
            magnitude=magnitude.astype(np.float32),
            metadata={"angle": angle.astype(np.float32)},
        )


def _to_grayscale(frame: Any, cv2) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        grayscale = array
    elif array.ndim == 3 and array.shape[2] >= 3:
        grayscale = cv2.cvtColor(array[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Expected a grayscale or RGB frame for optical flow estimation.")
    return grayscale.astype(np.uint8, copy=False)
