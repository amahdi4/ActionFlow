from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from flow.base import FlowEstimator, FlowField

logger = logging.getLogger(__name__)

MAX_EVENT_HISTORY = 12
ACTION_HISTORY = 8
DEFAULT_ANALYSIS_MAX_SIDE = 256
MOTION_THRESHOLD = 0.08
MIN_BACKGROUND_WARMUP_FRAMES = 4
PIL_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
MAX_BACKGROUND_MODEL_CACHE = 64
BACKGROUND_MODEL_CACHE: dict[str, Any | None] = {}


def create_live_state(backend: str) -> dict[str, Any]:
    session_id = uuid.uuid4().hex
    _set_background_model(session_id, _create_background_model())
    return {
        "session_id": session_id,
        "backend": backend,
        "frame_index": 0,
        "previous_frame": None,
        "previous_gray": None,
        "background_frames": 0,
        "last_timestamp": None,
        "smoothed_motion": 0.0,
        "last_status": "",
        "events": [],
        "feature_history": [],
        "analysis_failures": 0,
    }


def reset_live_outputs(
    backend: str,
    runtime_summary: Mapping[str, Any],
    backend_summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[None, str, str, dict[str, Any], str, dict[str, Any]]:
    state = create_live_state(backend)
    backend_summary = dict(backend_summaries.get(backend, {}))
    details = {
        "status": "waiting_for_camera",
        "selected_backend": backend,
        "backend_summary": backend_summary,
        "runtime": runtime_summary.get("runtime", {}),
        "event_history": state["events"],
    }
    return (
        None,
        _format_status_markdown(
            "Waiting for live camera frames",
            "Allow webcam access, then hold still for one second so ActionFlow can build a motion baseline.",
            {
                "backend": backend,
                "backend_available": backend_summary.get("available", False),
                "analysis_mode": "idle",
            },
        ),
        _format_guidance_markdown(
            "Stand where your head, shoulders, hands, and torso are visible.",
            "Once the scene is stable, perform one clear action for 2-3 seconds. Good first tests are: wave, arm raise, squat, side step, or sit-to-stand.",
        ),
        {
            "motion_energy": 0.0,
            "active_pixels": 0.0,
            "brightness": None,
            "contrast": None,
            "sharpness": None,
            "camera_shift_px": 0.0,
            "foreground_ratio": 0.0,
            "adaptive_threshold": MOTION_THRESHOLD,
            "backend": backend,
            "analysis_mode": "idle",
            "action_hypothesis": "idle",
            "action_confidence": 1.0,
        },
        json.dumps(details, indent=2, default=str),
        state,
    )


def analyze_live_frame(
    frame: np.ndarray | None,
    backend: str,
    state: dict[str, Any] | None,
    estimators: Mapping[str, FlowEstimator],
    backend_summaries: Mapping[str, Mapping[str, Any]],
    runtime_summary: Mapping[str, Any],
) -> tuple[np.ndarray | None, str, str, dict[str, Any], str, dict[str, Any]]:
    current_state = _normalize_state(state, backend)
    background_model = _get_background_model(current_state)
    if frame is None:
        logger.debug("Live analyzer received an empty frame for backend=%s", backend)
        return reset_live_outputs(backend, runtime_summary, backend_summaries)

    started_at = time.perf_counter()
    frame_rgb = _ensure_rgb(frame)
    analysis_frame = _prepare_analysis_frame(frame_rgb)
    frame_summary = _summarize_frame(analysis_frame)
    backend_summary = dict(backend_summaries.get(backend, {}))

    if current_state["previous_frame"] is None:
        pair_result = _empty_pair_result(frame_summary["gray"], backend, backend_summary)
    else:
        pair_result = _analyze_frame_pair(
            current_state["previous_frame"],
            current_state["previous_gray"],
            analysis_frame,
            frame_summary["gray"],
            backend,
            estimators,
            backend_summary,
            background_model=background_model,
            background_frames=int(current_state.get("background_frames", 0)),
        )

    current_state["frame_index"] += 1
    current_state["background_frames"] = int(current_state.get("background_frames", 0)) + 1
    current_state["smoothed_motion"] = (
        (current_state["smoothed_motion"] * 0.65) + (pair_result["motion_score"] * 0.35)
    )
    _append_feature_history(current_state, pair_result["features"])
    action_probe = _infer_action_hypothesis(current_state["feature_history"])

    fps_estimate = _estimate_fps(current_state["last_timestamp"])
    headline, guidance, camera_task = _build_live_feedback(
        frame_index=current_state["frame_index"],
        brightness=frame_summary["brightness"],
        contrast=frame_summary["contrast"],
        motion_score=pair_result["motion_score"],
        smoothed_motion=current_state["smoothed_motion"],
        active_ratio=pair_result["active_ratio"],
        camera_shift_px=pair_result["camera_shift_px"],
        backend_available=bool(backend_summary.get("available", False)),
        analysis_mode=pair_result["analysis_mode"],
        action_probe=action_probe,
    )

    processing_ms = (time.perf_counter() - started_at) * 1000.0
    metrics = {
        "motion_energy": round(pair_result["motion_score"], 4),
        "smoothed_motion": round(current_state["smoothed_motion"], 4),
        "active_pixels": round(pair_result["active_ratio"], 4),
        "brightness": round(frame_summary["brightness"], 4),
        "contrast": round(frame_summary["contrast"], 4),
        "sharpness": round(frame_summary["sharpness"], 4),
        "camera_shift_px": round(pair_result["camera_shift_px"], 3),
        "foreground_ratio": round(pair_result["foreground_ratio"], 4),
        "adaptive_threshold": round(pair_result["adaptive_threshold"], 4),
        "fps_estimate": round(fps_estimate, 2) if fps_estimate is not None else None,
        "processing_ms": round(processing_ms, 2),
        "backend": backend,
        "backend_available": backend_summary.get("available", False),
        "analysis_mode": pair_result["analysis_mode"],
        "action_hypothesis": action_probe["display_label"],
        "action_confidence": round(action_probe["confidence"], 4),
        "action_scores": action_probe["scores"],
    }
    overlay = _render_analysis_frame(
        frame_rgb,
        pair_result["motion_map"],
        headline=headline,
        subtitle=(
            f"backend={backend} | mode={pair_result['analysis_mode']} | "
            f"motion={metrics['motion_energy']:.4f} | active={metrics['active_pixels']:.2%} | "
            f"shift={metrics['camera_shift_px']:.2f}px"
        ),
    )

    event = {
        "frame_index": current_state["frame_index"],
        "status": headline,
        "motion_energy": metrics["motion_energy"],
        "active_pixels": metrics["active_pixels"],
        "brightness": metrics["brightness"],
        "analysis_mode": pair_result["analysis_mode"],
        "processing_ms": metrics["processing_ms"],
        "camera_shift_px": metrics["camera_shift_px"],
        "action_hypothesis": action_probe["display_label"],
        "action_confidence": metrics["action_confidence"],
    }
    _append_event(current_state, event)
    _maybe_log_event(current_state, event)

    details = {
        "status": headline,
        "guidance": guidance,
        "camera_task": camera_task,
        "selected_backend": backend,
        "backend_summary": backend_summary,
        "runtime": runtime_summary.get("runtime", {}),
        "frame_shape": list(frame_rgb.shape),
        "analysis_shape": list(analysis_frame.shape),
        "frame_index": current_state["frame_index"],
        "metrics": metrics,
        "action_probe": action_probe,
        "analysis_notes": pair_result["notes"],
        "event_history": current_state["events"],
    }

    current_state["previous_frame"] = analysis_frame
    current_state["previous_gray"] = frame_summary["gray"]
    current_state["last_timestamp"] = time.time()
    current_state["last_status"] = headline
    if pair_result["analysis_mode"].endswith("fallback"):
        current_state["analysis_failures"] += 1

    return (
        overlay,
        _format_status_markdown(headline, guidance, metrics, action_probe),
        _format_guidance_markdown(camera_task, _research_note(backend_summary, pair_result["analysis_mode"])),
        metrics,
        json.dumps(details, indent=2, default=str),
        current_state,
    )


def analyze_uploaded_video(
    video_path: str | None,
    backend: str,
    estimators: Mapping[str, FlowEstimator],
    backend_summaries: Mapping[str, Mapping[str, Any]],
    runtime_summary: Mapping[str, Any],
) -> tuple[np.ndarray | None, str, str, dict[str, Any], str]:
    backend_summary = dict(backend_summaries.get(backend, {}))
    if video_path is None:
        details = {
            "status": "waiting_for_upload",
            "selected_backend": backend,
            "backend_summary": backend_summary,
            "runtime": runtime_summary.get("runtime", {}),
        }
        return (
            None,
            _format_status_markdown(
                "Upload a clip to review it",
                "ActionFlow will scan the clip for motion strength, scene quality, and backend availability.",
                {
                    "backend": backend,
                    "backend_available": backend_summary.get("available", False),
                    "analysis_mode": "idle",
                },
            ),
            _format_guidance_markdown(
                "Use short clips with one action per clip.",
                "For action-recognition data collection, keep the full body or upper body visible and avoid camera movement.",
            ),
            {"backend": backend, "analysis_mode": "idle"},
            json.dumps(details, indent=2, default=str),
        )

    try:
        import cv2
    except ModuleNotFoundError:
        details = {
            "status": "opencv_missing",
            "selected_backend": backend,
            "video_file": Path(video_path).name,
            "runtime": runtime_summary.get("runtime", {}),
        }
        return (
            None,
            _format_status_markdown(
                "Video review requires OpenCV",
                "Install `opencv-python` to decode uploaded videos and compute clip-level motion summaries.",
                {
                    "backend": backend,
                    "backend_available": backend_summary.get("available", False),
                    "analysis_mode": "opencv_missing",
                },
            ),
            _format_guidance_markdown(
                "The live camera tab still works with frame-difference fallback.",
                "For uploaded clips, install the full runtime dependencies so the backend can inspect the video file.",
            ),
            {"backend": backend, "analysis_mode": "opencv_missing"},
            json.dumps(details, indent=2, default=str),
        )

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        details = {
            "status": "video_open_failed",
            "selected_backend": backend,
            "video_file": Path(video_path).name,
        }
        return (
            None,
            _format_status_markdown(
                "Could not open the uploaded clip",
                "The file was received but OpenCV could not decode it.",
                {"backend": backend, "backend_available": backend_summary.get("available", False)},
            ),
            _format_guidance_markdown(
                "Try exporting the clip as MP4 with H.264 encoding.",
                "Short, stable clips are easier to inspect and later easier to train on.",
            ),
            {"backend": backend, "analysis_mode": "video_open_failed"},
            json.dumps(details, indent=2, default=str),
        )

    logger.info("Reviewing uploaded clip | backend=%s | file=%s", backend, Path(video_path).name)
    previous_frame: np.ndarray | None = None
    previous_gray: np.ndarray | None = None
    sampled_pairs = 0
    total_frames = 0
    background_model = _create_background_model()
    metrics_history: list[dict[str, float]] = []
    feature_history: list[dict[str, Any]] = []
    peak_result: dict[str, Any] | None = None
    peak_frame: np.ndarray | None = None

    while sampled_pairs < 48:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        total_frames += 1
        if total_frames % 2 == 1:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        analysis_frame = _prepare_analysis_frame(frame_rgb)
        frame_summary = _summarize_frame(analysis_frame)
        if previous_frame is not None and previous_gray is not None:
            pair_result = _analyze_frame_pair(
                previous_frame,
                previous_gray,
                analysis_frame,
                frame_summary["gray"],
                backend,
                estimators,
                backend_summary,
                background_model=background_model,
                background_frames=sampled_pairs,
            )
            metrics_history.append(
                {
                    "motion_energy": pair_result["motion_score"],
                    "active_pixels": pair_result["active_ratio"],
                    "brightness": frame_summary["brightness"],
                    "contrast": frame_summary["contrast"],
                    "sharpness": frame_summary["sharpness"],
                    "camera_shift_px": pair_result["camera_shift_px"],
                    "foreground_ratio": pair_result["foreground_ratio"],
                }
            )
            feature_history.append(pair_result["features"])
            if peak_result is None or pair_result["motion_score"] > peak_result["motion_score"]:
                peak_result = pair_result
                peak_frame = frame_rgb
            sampled_pairs += 1
        previous_frame = analysis_frame
        previous_gray = frame_summary["gray"]

    capture.release()

    if not metrics_history or peak_result is None or peak_frame is None:
        details = {
            "status": "insufficient_video_frames",
            "selected_backend": backend,
            "video_file": Path(video_path).name,
        }
        return (
            None,
            _format_status_markdown(
                "Not enough decoded frames to review the clip",
                "The video loaded, but there were too few frames to compute motion metrics.",
                {"backend": backend, "backend_available": backend_summary.get("available", False)},
            ),
            _format_guidance_markdown(
                "Upload a slightly longer clip.",
                "Two to five seconds is a good range for quick manual review.",
            ),
            {"backend": backend, "analysis_mode": "insufficient_video_frames"},
            json.dumps(details, indent=2, default=str),
        )

    motion_values = np.array([item["motion_energy"] for item in metrics_history], dtype=np.float32)
    active_values = np.array([item["active_pixels"] for item in metrics_history], dtype=np.float32)
    brightness_values = np.array([item["brightness"] for item in metrics_history], dtype=np.float32)
    contrast_values = np.array([item["contrast"] for item in metrics_history], dtype=np.float32)
    sharpness_values = np.array([item["sharpness"] for item in metrics_history], dtype=np.float32)
    camera_shift_values = np.array([item["camera_shift_px"] for item in metrics_history], dtype=np.float32)
    foreground_values = np.array([item["foreground_ratio"] for item in metrics_history], dtype=np.float32)
    action_probe = _infer_action_hypothesis(feature_history)
    headline, guidance, camera_task = _build_clip_feedback(
        mean_motion=float(motion_values.mean()),
        peak_motion=float(motion_values.max()),
        active_ratio=float(active_values.mean()),
        mean_camera_shift=float(camera_shift_values.mean()),
        brightness=float(brightness_values.mean()),
        contrast=float(contrast_values.mean()),
        action_probe=action_probe,
    )
    metrics = {
        "backend": backend,
        "analysis_mode": peak_result["analysis_mode"],
        "frames_reviewed": len(metrics_history),
        "mean_motion_energy": round(float(motion_values.mean()), 4),
        "peak_motion_energy": round(float(motion_values.max()), 4),
        "mean_active_pixels": round(float(active_values.mean()), 4),
        "mean_brightness": round(float(brightness_values.mean()), 4),
        "mean_contrast": round(float(contrast_values.mean()), 4),
        "mean_sharpness": round(float(sharpness_values.mean()), 4),
        "mean_camera_shift_px": round(float(camera_shift_values.mean()), 3),
        "mean_foreground_ratio": round(float(foreground_values.mean()), 4),
        "backend_available": backend_summary.get("available", False),
        "action_hypothesis": action_probe["display_label"],
        "action_confidence": round(action_probe["confidence"], 4),
        "action_scores": action_probe["scores"],
    }
    overlay = _render_analysis_frame(
        peak_frame,
        peak_result["motion_map"],
        headline=headline,
        subtitle=(
            f"backend={backend} | mode={peak_result['analysis_mode']} | "
            f"peak_motion={metrics['peak_motion_energy']:.4f} | avg_motion={metrics['mean_motion_energy']:.4f} | "
            f"shift={metrics['mean_camera_shift_px']:.2f}px"
        ),
    )
    details = {
        "status": headline,
        "guidance": guidance,
        "clip_task": camera_task,
        "selected_backend": backend,
        "backend_summary": backend_summary,
        "runtime": runtime_summary.get("runtime", {}),
        "video_file": Path(video_path).name,
        "metrics": metrics,
        "action_probe": action_probe,
        "analysis_notes": peak_result["notes"],
    }
    logger.info(
        "Finished uploaded clip review | file=%s | backend=%s | mean_motion=%.4f | peak_motion=%.4f",
        Path(video_path).name,
        backend,
        float(motion_values.mean()),
        float(motion_values.max()),
    )
    return (
        overlay,
        _format_status_markdown(headline, guidance, metrics, action_probe),
        _format_guidance_markdown(camera_task, _research_note(backend_summary, peak_result["analysis_mode"])),
        metrics,
        json.dumps(details, indent=2, default=str),
    )


def _normalize_state(state: dict[str, Any] | None, backend: str) -> dict[str, Any]:
    if state is None:
        return create_live_state(backend)
    if state.get("backend") != backend:
        logger.info("Resetting live analyzer state because backend changed to %s", backend)
        return create_live_state(backend)
    session_id = state.get("session_id")
    if not session_id:
        state = dict(state)
        state["session_id"] = uuid.uuid4().hex
    return state


def _get_background_model(state: Mapping[str, Any]) -> Any | None:
    session_id = str(state.get("session_id", ""))
    if not session_id:
        return None
    if session_id not in BACKGROUND_MODEL_CACHE:
        _set_background_model(session_id, _create_background_model())
    return BACKGROUND_MODEL_CACHE.get(session_id)


def _set_background_model(session_id: str, background_model: Any | None) -> None:
    BACKGROUND_MODEL_CACHE[session_id] = background_model
    while len(BACKGROUND_MODEL_CACHE) > MAX_BACKGROUND_MODEL_CACHE:
        oldest_session_id = next(iter(BACKGROUND_MODEL_CACHE))
        if oldest_session_id == session_id and len(BACKGROUND_MODEL_CACHE) == 1:
            break
        BACKGROUND_MODEL_CACHE.pop(oldest_session_id, None)


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[..., :3]
    elif array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Expected an RGB webcam frame.")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array[..., :3])


def _prepare_analysis_frame(frame_rgb: np.ndarray, *, max_side: int = DEFAULT_ANALYSIS_MAX_SIDE) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return frame_rgb

    scale = max_side / float(longest_side)
    resized_width = max(32, int(round(width * scale)))
    resized_height = max(32, int(round(height * scale)))
    cv2 = _import_cv2()
    if cv2 is not None:
        resized = cv2.resize(frame_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    else:
        resized = np.asarray(Image.fromarray(frame_rgb).resize((resized_width, resized_height), resample=PIL_BILINEAR))
    return np.ascontiguousarray(resized)


def _summarize_frame(frame_rgb: np.ndarray) -> dict[str, Any]:
    gray = _to_grayscale(frame_rgb)
    vertical_grad = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    horizontal_grad = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    sharpness = float((vertical_grad + horizontal_grad) / 255.0)
    return {
        "gray": gray,
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "sharpness": sharpness,
    }


def _create_background_model() -> Any | None:
    cv2 = _import_cv2()
    if cv2 is None:
        return None
    return cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=32, detectShadows=False)


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError:
        return None
    return cv2


def _extract_motion_features(motion_map: np.ndarray, flow_field: FlowField | None) -> dict[str, Any]:
    weights = np.asarray(motion_map, dtype=np.float32)
    total = float(weights.sum())
    height, width = weights.shape
    y_coords, x_coords = np.indices(weights.shape, dtype=np.float32)
    denominator_x = max(width - 1, 1)
    denominator_y = max(height - 1, 1)
    if total <= 1e-6:
        centroid_x = 0.5
        centroid_y = 0.5
        top_ratio = 0.5
        bottom_ratio = 0.5
        left_ratio = 0.5
        right_ratio = 0.5
    else:
        centroid_x = float((weights * x_coords).sum() / total / denominator_x)
        centroid_y = float((weights * y_coords).sum() / total / denominator_y)
        top_ratio = float(weights[: height // 2].sum() / total)
        bottom_ratio = float(weights[height // 2 :].sum() / total)
        left_ratio = float(weights[:, : width // 2].sum() / total)
        right_ratio = float(weights[:, width // 2 :].sum() / total)

    directional_available = False
    horizontal_energy = 0.0
    vertical_energy = 0.0
    horizontal_dominance = 0.0
    vertical_dominance = 0.0
    signed_h = 0.0
    signed_v = 0.0
    if flow_field is not None and flow_field.horizontal is not None and flow_field.vertical is not None:
        horizontal = np.asarray(flow_field.horizontal, dtype=np.float32)
        vertical = np.asarray(flow_field.vertical, dtype=np.float32)
        if horizontal.shape == weights.shape and vertical.shape == weights.shape and total > 1e-6:
            directional_available = True
            horizontal_energy = float((weights * np.abs(horizontal)).sum() / total)
            vertical_energy = float((weights * np.abs(vertical)).sum() / total)
            total_directional = horizontal_energy + vertical_energy
            if total_directional > 1e-6:
                horizontal_dominance = horizontal_energy / total_directional
                vertical_dominance = vertical_energy / total_directional
            signed_h = float((weights * horizontal).sum() / total)
            signed_v = float((weights * vertical).sum() / total)

    return {
        "directional_available": directional_available,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "top_ratio": top_ratio,
        "bottom_ratio": bottom_ratio,
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
        "horizontal_energy": horizontal_energy,
        "vertical_energy": vertical_energy,
        "horizontal_dominance": horizontal_dominance,
        "vertical_dominance": vertical_dominance,
        "signed_h": signed_h,
        "signed_v": signed_v,
    }


def _analyze_frame_pair(
    previous_frame: np.ndarray,
    previous_gray: np.ndarray,
    current_frame: np.ndarray,
    current_gray: np.ndarray,
    backend: str,
    estimators: Mapping[str, FlowEstimator],
    backend_summary: Mapping[str, Any],
    *,
    background_model: Any | None = None,
    background_frames: int = 0,
) -> dict[str, Any]:
    camera_motion = _estimate_camera_motion(previous_gray, current_gray)
    aligned_frame = current_frame
    aligned_gray = current_gray
    if camera_motion["reliable"] and camera_motion["magnitude"] >= 0.25:
        aligned_frame = _translate_frame(current_frame, -camera_motion["dx"], -camera_motion["dy"])
        aligned_gray = _translate_frame(current_gray, -camera_motion["dx"], -camera_motion["dy"])

    difference_map = np.abs(aligned_gray - previous_gray) / 255.0
    motion_map = difference_map
    analysis_mode = "frame_difference_fallback"
    notes: list[str] = []
    flow_field: FlowField | None = None
    estimator = estimators.get(backend)
    if estimator is not None and backend_summary.get("available", False):
        try:
            flow_field = estimator.estimate(previous_frame, aligned_frame)
            flow_motion = _flow_to_motion_map(flow_field)
            if flow_motion is not None:
                motion_map = np.clip((flow_motion * 0.72) + (difference_map * 0.28), 0.0, 1.0)
                analysis_mode = f"{backend}_optical_flow"
                track_count = flow_field.metadata.get("num_tracks")
                if track_count is not None:
                    notes.append(f"tracked_features={track_count}")
        except NotImplementedError as exc:
            notes.append(str(exc))
            logger.warning("Backend %s is not fully implemented; using frame-difference fallback.", backend)
        except Exception:
            notes.append("Optical flow estimation failed; using frame-difference fallback.")
            logger.exception("Optical flow estimation failed for backend=%s", backend)
    else:
        details = backend_summary.get("availability_details")
        if details:
            notes.append(details)

    foreground_mask = _foreground_motion_mask(aligned_frame, background_model)
    foreground_ratio = float(foreground_mask.mean()) if foreground_mask is not None else 0.0
    if foreground_mask is not None and background_frames >= MIN_BACKGROUND_WARMUP_FRAMES:
        motion_map = np.clip(np.maximum(motion_map, foreground_mask * 0.7), 0.0, 1.0)
        notes.append("foreground_model=mog2")

    motion_map = _smooth_motion_map(motion_map)
    adaptive_threshold = _adaptive_motion_threshold(motion_map)
    salient_motion = _salient_motion_map(motion_map, adaptive_threshold)
    motion_score = float(salient_motion.mean())
    active_ratio = float(np.mean(salient_motion > max(MOTION_THRESHOLD, adaptive_threshold)))
    features = _extract_motion_features(salient_motion, flow_field)
    features["motion_score"] = motion_score
    features["active_ratio"] = active_ratio
    features["camera_shift_px"] = camera_motion["magnitude"]
    features["camera_motion_response"] = camera_motion["response"]
    features["foreground_ratio"] = foreground_ratio
    return {
        "motion_map": salient_motion,
        "motion_score": motion_score,
        "active_ratio": active_ratio,
        "adaptive_threshold": adaptive_threshold,
        "foreground_ratio": foreground_ratio,
        "camera_shift_px": camera_motion["magnitude"],
        "analysis_mode": analysis_mode,
        "notes": notes,
        "features": features,
    }


def _empty_pair_result(
    current_gray: np.ndarray,
    backend: str,
    backend_summary: Mapping[str, Any],
) -> dict[str, Any]:
    notes = ["Waiting for a second frame to establish motion."]
    if not backend_summary.get("available", False):
        details = backend_summary.get("availability_details")
        if details:
            notes.append(details)
    return {
        "motion_map": np.zeros_like(current_gray, dtype=np.float32),
        "motion_score": 0.0,
        "active_ratio": 0.0,
        "adaptive_threshold": MOTION_THRESHOLD,
        "foreground_ratio": 0.0,
        "camera_shift_px": 0.0,
        "analysis_mode": f"{backend}_warmup",
        "notes": notes,
        "features": {
            "motion_score": 0.0,
            "active_ratio": 0.0,
            "directional_available": False,
            "centroid_x": 0.5,
            "centroid_y": 0.5,
            "top_ratio": 0.5,
            "bottom_ratio": 0.5,
            "left_ratio": 0.5,
            "right_ratio": 0.5,
            "horizontal_energy": 0.0,
            "vertical_energy": 0.0,
            "horizontal_dominance": 0.0,
            "vertical_dominance": 0.0,
            "signed_h": 0.0,
            "signed_v": 0.0,
            "camera_shift_px": 0.0,
            "camera_motion_response": 0.0,
            "foreground_ratio": 0.0,
        },
    }


def _flow_to_motion_map(flow_field: FlowField) -> np.ndarray | None:
    if flow_field.magnitude is None:
        return None
    magnitude = np.asarray(flow_field.magnitude, dtype=np.float32)
    if magnitude.size == 0:
        return None
    percentile = float(np.percentile(magnitude, 98))
    scale = percentile if percentile > 1e-6 else float(magnitude.max())
    if scale <= 1e-6:
        return np.zeros_like(magnitude, dtype=np.float32)
    return np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)


def _to_grayscale(frame_rgb: np.ndarray) -> np.ndarray:
    return frame_rgb.astype(np.float32).mean(axis=2)


def _estimate_camera_motion(previous_gray: np.ndarray, current_gray: np.ndarray) -> dict[str, float | bool]:
    cv2 = _import_cv2()
    if cv2 is None or previous_gray.shape != current_gray.shape:
        return {"dx": 0.0, "dy": 0.0, "magnitude": 0.0, "response": 0.0, "reliable": False}

    if max(float(np.std(previous_gray)), float(np.std(current_gray))) < 4.0:
        return {"dx": 0.0, "dy": 0.0, "magnitude": 0.0, "response": 0.0, "reliable": False}

    try:
        shift, response = cv2.phaseCorrelate(
            np.asarray(previous_gray, dtype=np.float32),
            np.asarray(current_gray, dtype=np.float32),
        )
    except Exception:
        logger.debug("Camera-motion estimation failed; continuing without compensation.", exc_info=True)
        return {"dx": 0.0, "dy": 0.0, "magnitude": 0.0, "response": 0.0, "reliable": False}

    dx = float(shift[0])
    dy = float(shift[1])
    magnitude = float((dx * dx + dy * dy) ** 0.5)
    response = float(response)
    reliable = response >= 0.02 and magnitude <= (max(previous_gray.shape) * 0.2)
    return {
        "dx": dx,
        "dy": dy,
        "magnitude": magnitude,
        "response": response,
        "reliable": reliable,
    }


def _translate_frame(frame: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    cv2 = _import_cv2()
    if cv2 is None:
        return frame

    height, width = frame.shape[:2]
    transform = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _foreground_motion_mask(frame_rgb: np.ndarray, background_model: Any | None) -> np.ndarray | None:
    cv2 = _import_cv2()
    if cv2 is None or background_model is None:
        return None

    try:
        foreground_mask = background_model.apply(frame_rgb, learningRate=0.02)
        foreground_mask = cv2.medianBlur(foreground_mask, 5)
        kernel = np.ones((3, 3), dtype=np.uint8)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    except Exception:
        logger.debug("Foreground-model update failed; continuing without it.", exc_info=True)
        return None
    return np.clip(foreground_mask.astype(np.float32) / 255.0, 0.0, 1.0)


def _smooth_motion_map(motion_map: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    motion_map = np.asarray(motion_map, dtype=np.float32)
    if cv2 is not None:
        return cv2.GaussianBlur(motion_map, (5, 5), 0)

    padded = np.pad(motion_map, ((1, 1), (1, 1)), mode="edge")
    neighborhoods = [
        padded[0:-2, 0:-2],
        padded[0:-2, 1:-1],
        padded[0:-2, 2:],
        padded[1:-1, 0:-2],
        padded[1:-1, 1:-1],
        padded[1:-1, 2:],
        padded[2:, 0:-2],
        padded[2:, 1:-1],
        padded[2:, 2:],
    ]
    return sum(neighborhoods) / float(len(neighborhoods))


def _adaptive_motion_threshold(motion_map: np.ndarray) -> float:
    mean_value = float(np.mean(motion_map))
    std_value = float(np.std(motion_map))
    upper_tail = float(np.percentile(motion_map, 90))
    return float(
        max(
            0.05,
            min(
                0.35,
                max((mean_value + (0.85 * std_value)), (upper_tail * 0.35), MOTION_THRESHOLD),
            ),
        )
    )


def _salient_motion_map(motion_map: np.ndarray, adaptive_threshold: float) -> np.ndarray:
    floor = adaptive_threshold * 0.45
    return np.clip((motion_map - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0).astype(np.float32)


def _estimate_fps(last_timestamp: float | None) -> float | None:
    if last_timestamp is None:
        return None
    delta = time.time() - last_timestamp
    if delta <= 1e-6:
        return None
    return 1.0 / delta


def _append_feature_history(state: dict[str, Any], features: dict[str, Any]) -> None:
    history = list(state.get("feature_history", []))
    history.append(features)
    state["feature_history"] = history[-ACTION_HISTORY:]


def _infer_action_hypothesis(feature_history: list[dict[str, Any]]) -> dict[str, Any]:
    if not feature_history:
        return _action_probe("idle", 1.0, {"idle": 1.0}, "Waiting for enough frames to infer motion.")

    recent = feature_history[-ACTION_HISTORY:]
    mean_motion = float(np.mean([item["motion_score"] for item in recent]))
    max_motion = float(np.max([item["motion_score"] for item in recent]))
    mean_active = float(np.mean([item["active_ratio"] for item in recent]))
    directional = any(item["directional_available"] for item in recent)
    avg_horizontal_dom = float(np.mean([item["horizontal_dominance"] for item in recent]))
    avg_vertical_dom = float(np.mean([item["vertical_dominance"] for item in recent]))
    avg_top_ratio = float(np.mean([item["top_ratio"] for item in recent]))
    avg_bottom_ratio = float(np.mean([item["bottom_ratio"] for item in recent]))
    avg_centroid_y = float(np.mean([item["centroid_y"] for item in recent]))
    avg_signed_h = float(np.mean([item["signed_h"] for item in recent]))
    avg_signed_v = float(np.mean([item["signed_v"] for item in recent]))
    avg_camera_shift = float(np.mean([item.get("camera_shift_px", 0.0) for item in recent]))
    avg_camera_response = float(np.mean([item.get("camera_motion_response", 0.0) for item in recent]))
    avg_foreground_ratio = float(np.mean([item.get("foreground_ratio", 0.0) for item in recent]))
    horizontal_oscillation = _oscillation_strength([item["signed_h"] for item in recent])
    vertical_oscillation = _oscillation_strength([item["signed_v"] for item in recent])

    if mean_motion < 0.03 and avg_camera_shift < 1.4:
        confidence = _clamp01((0.05 - mean_motion) / 0.05)
        return _action_probe(
            "idle",
            max(confidence, 0.65),
            {"idle": max(confidence, 0.65)},
            "Scene is stable and ready for a clean action take.",
        )

    high_motion = _clamp01((max_motion - 0.08) / 0.18)
    whole_body_motion = _clamp01((mean_active - 0.12) / 0.28)
    localized_motion = 1.0 - _clamp01((mean_active - 0.3) / 0.3)
    top_focus = _clamp01((avg_top_ratio - 0.52) / 0.35)
    bottom_focus = _clamp01((avg_bottom_ratio - 0.5) / 0.35)
    upper_body_focus = _clamp01((0.62 - avg_centroid_y) / 0.32)
    lower_body_focus = _clamp01((avg_centroid_y - 0.48) / 0.28)
    horizontal_dom = _clamp01((avg_horizontal_dom - 0.52) / 0.35)
    vertical_dom = _clamp01((avg_vertical_dom - 0.52) / 0.35)
    upward_motion = _clamp01((-avg_signed_v - 0.02) / 0.08)
    downward_motion = _clamp01((avg_signed_v - 0.02) / 0.08)
    lateral_consistency = _clamp01((abs(avg_signed_h) - 0.015) / 0.08)
    global_shift = _clamp01((avg_camera_shift - 1.2) / 5.5)
    camera_response = _clamp01((avg_camera_response - 0.05) / 0.2)
    foreground_support = _clamp01((avg_foreground_ratio - 0.04) / 0.18)

    scores: dict[str, float] = {
        "camera_motion": (
            (0.36 * global_shift)
            + (0.28 * _clamp01((mean_active - 0.34) / 0.24))
            + (0.18 * camera_response)
            + (0.18 * (1.0 - foreground_support))
        ),
        "general_motion": (0.45 * high_motion) + (0.35 * _clamp01((mean_active - 0.08) / 0.2)) + (0.2 * foreground_support),
        "wave": 0.0,
        "arm_raise": 0.0,
        "side_step": 0.0,
        "squat_like": 0.0,
        "jumping_like": 0.0,
    }
    if directional:
        scores["wave"] = (
            (0.28 * horizontal_dom)
            + (0.2 * top_focus)
            + (0.16 * upper_body_focus)
            + (0.2 * horizontal_oscillation)
            + (0.16 * localized_motion)
        )
        scores["arm_raise"] = (
            (0.28 * vertical_dom)
            + (0.22 * top_focus)
            + (0.16 * upper_body_focus)
            + (0.2 * upward_motion)
            + (0.14 * localized_motion)
        )
        scores["side_step"] = (
            (0.28 * horizontal_dom)
            + (0.18 * whole_body_motion)
            + (0.18 * lateral_consistency)
            + (0.2 * lower_body_focus)
            + (0.16 * high_motion)
        )
        scores["squat_like"] = (
            (0.24 * vertical_dom)
            + (0.22 * bottom_focus)
            + (0.16 * downward_motion)
            + (0.2 * whole_body_motion)
            + (0.18 * vertical_oscillation)
        )
        scores["jumping_like"] = (
            (0.22 * vertical_dom)
            + (0.22 * whole_body_motion)
            + (0.22 * high_motion)
            + (0.18 * vertical_oscillation)
            + (0.16 * _clamp01((0.2 - abs(avg_signed_v)) / 0.2))
        )

    if scores["camera_motion"] > 0.6 or (avg_camera_shift > 2.2 and mean_motion < 0.14):
        return _action_probe(
            "camera_motion",
            scores["camera_motion"],
            scores,
            "The dominant motion looks global rather than body-localized, which suggests camera shake or background movement.",
        )

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    if best_score < 0.34:
        return _action_probe(
            "general_motion",
            max(scores["general_motion"], 0.35),
            scores,
            "Motion is present, but the pattern is not distinct enough yet for a more specific heuristic label.",
        )

    explanations = {
        "wave": "Alternating horizontal motion is concentrated in the upper body.",
        "arm_raise": "Vertical motion is concentrated in the upper body and trends upward.",
        "side_step": "Lateral motion involves more of the body and stays consistent over several frames.",
        "squat_like": "Vertical motion is stronger in the lower half of the frame and resembles a squat or sit-to-stand.",
        "jumping_like": "Strong whole-body vertical motion is present across multiple frames.",
        "general_motion": "Motion is present, but it does not cleanly match one of the supported heuristic action templates.",
    }
    return _action_probe(best_label, best_score, scores, explanations[best_label])


def _action_probe(
    label: str,
    confidence: float,
    raw_scores: Mapping[str, float],
    explanation: str,
) -> dict[str, Any]:
    sorted_scores = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
    top_scores = { _display_label(name): round(float(score), 4) for name, score in sorted_scores[:4] if score > 0.02 }
    return {
        "label": label,
        "display_label": _display_label(label),
        "confidence": round(float(confidence), 4),
        "scores": top_scores,
        "explanation": explanation,
    }


def _display_label(label: str) -> str:
    mapping = {
        "idle": "Idle / Ready",
        "camera_motion": "Camera Motion",
        "general_motion": "General Motion",
        "wave": "Wave",
        "arm_raise": "Arm Raise",
        "side_step": "Side Step",
        "squat_like": "Squat / Sit-to-Stand",
        "jumping_like": "Jumping Motion",
    }
    return mapping.get(label, label.replace("_", " ").title())


def _oscillation_strength(values: list[float]) -> float:
    filtered = [value for value in values if abs(value) > 1e-4]
    if len(filtered) < 3:
        return 0.0
    sign_changes = 0
    for left, right in zip(filtered, filtered[1:]):
        if left == 0 or right == 0:
            continue
        if (left > 0) != (right > 0):
            sign_changes += 1
    return _clamp01(sign_changes / 3.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _build_live_feedback(
    *,
    frame_index: int,
    brightness: float,
    contrast: float,
    motion_score: float,
    smoothed_motion: float,
    active_ratio: float,
    camera_shift_px: float,
    backend_available: bool,
    analysis_mode: str,
    action_probe: Mapping[str, Any],
) -> tuple[str, str, str]:
    if frame_index <= 1:
        return (
            "Building a motion baseline",
            "Keep still for one second so the app can compare the next frame against a stable reference.",
            "Stand in frame with your head, shoulders, elbows, and hands visible. Wait for the overlay to settle before moving.",
        )
    if brightness < 0.18:
        return (
            "Lighting is too low",
            "The camera image is too dark for reliable motion cues. Add front lighting or face a window.",
            "Improve lighting first. Once the scene is brighter, hold still briefly and then perform one clear action.",
        )
    if contrast < 0.05:
        return (
            "Scene contrast is too flat",
            "The foreground and background are blending together. Use a cleaner background or stronger subject lighting.",
            "Step away from the background or add side/front light so your limbs separate visually from the room.",
        )
    if action_probe["label"] == "camera_motion" or camera_shift_px > 2.0 or (active_ratio > 0.45 and motion_score > 0.16):
        return (
            "Action hypothesis: Camera Motion",
            "The dominant motion looks global rather than body-localized, which usually means the camera is moving or the background is noisy.",
            "Stabilize the laptop or phone and reduce moving background objects before collecting action data.",
        )
    if action_probe["label"] == "idle" or smoothed_motion < 0.03:
        return (
            "Ready for an action take",
            "The stream is stable and ready to capture a clean action segment.",
            "Start one action now and keep it clear for 2-3 seconds. Good tests are: wave, arm raise, squat, side step, or sit-to-stand.",
        )
    if action_probe["label"] not in {"idle", "general_motion", "camera_motion"}:
        return (
            f"Action hypothesis: {action_probe['display_label']}",
            f"{action_probe['explanation']} Heuristic confidence: {action_probe['confidence']:.2f}.",
            _action_collection_tip(action_probe["label"]),
        )
    message = "The stream is producing a usable motion signal."
    if not backend_available or analysis_mode.endswith("fallback"):
        message = (
            "The stream is usable, but the app is relying on frame-difference fallback rather than a full optical-flow backend."
        )
    return (
        "Action hypothesis: General Motion",
        message,
        "Continue the motion cleanly, keep your full movement inside the frame, then return to a still pose before the next take.",
    )


def _build_clip_feedback(
    *,
    mean_motion: float,
    peak_motion: float,
    active_ratio: float,
    mean_camera_shift: float,
    brightness: float,
    contrast: float,
    action_probe: Mapping[str, Any],
) -> tuple[str, str, str]:
    if brightness < 0.18:
        return (
            "Clip is too dark for reliable review",
            "The clip has low brightness, which will make downstream feature extraction unstable.",
            "Re-record the clip with stronger front lighting and keep the camera locked off.",
        )
    if mean_motion < 0.02 and peak_motion < 0.05:
        return (
            "Clip has very little motion",
            "There is not enough visible movement for an action-recognition sample.",
            "Re-record with a clearer action or trim the clip so the action fills most of the duration.",
        )
    if mean_camera_shift > 1.8 or (active_ratio > 0.45 and peak_motion > 0.18):
        return (
            "Clip likely contains camera shake",
            "The motion is spread across most of the frame, which usually means camera motion rather than subject motion.",
            "Re-record with the camera fixed and the subject moving independently of the background.",
        )
    if contrast < 0.05:
        return (
            "Clip contrast is weak",
            "The clip is usable, but subject/background separation is poor.",
            "Use a simpler background or stronger lighting before collecting a larger dataset.",
        )
    if action_probe["label"] not in {"idle", "general_motion", "camera_motion"}:
        return (
            f"Clip hypothesis: {action_probe['display_label']}",
            f"{action_probe['explanation']} Heuristic confidence: {action_probe['confidence']:.2f}.",
            _action_collection_tip(action_probe["label"]),
        )
    return (
        "Clip review looks usable",
        "The clip has enough motion signal to be useful for manual inspection and early data collection.",
        "For the next recording, keep one action per clip and leave a short still segment before and after the action.",
    )


def _research_note(backend_summary: Mapping[str, Any], analysis_mode: str) -> str:
    if backend_summary.get("available", False) and analysis_mode.endswith("optical_flow"):
        return "The selected backend is active. Use this view to compare motion signatures before wiring a full model."
    return (
        "The app is falling back to a lighter motion heuristic. That is still useful for framing, lighting, and data-collection checks."
    )


def _render_analysis_frame(
    frame_rgb: np.ndarray,
    motion_map: np.ndarray,
    *,
    headline: str,
    subtitle: str,
) -> np.ndarray:
    motion_map = _resize_motion_map(motion_map, frame_rgb.shape[:2])
    motion_uint8 = np.clip(motion_map * 255.0, 0, 255).astype(np.uint8)
    heat_rgb = np.zeros_like(frame_rgb)
    heat_rgb[..., 0] = motion_uint8
    heat_rgb[..., 1] = (motion_uint8 * 0.42).astype(np.uint8)
    heat_rgb[..., 2] = (motion_uint8 * 0.14).astype(np.uint8)
    alpha = np.clip(motion_map[..., None] * 0.72, 0.0, 0.72)
    overlay = np.clip((frame_rgb * (1.0 - alpha)) + (heat_rgb * alpha) + (frame_rgb * 0.06), 0, 255).astype(np.uint8)

    canvas = Image.fromarray(overlay)
    width, _ = canvas.size
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 70), fill=(14, 18, 24))
    draw.text((16, 14), headline, fill=(245, 247, 250))
    draw.text((16, 40), subtitle, fill=(160, 175, 192))
    return np.asarray(canvas)


def _resize_motion_map(motion_map: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_shape
    if motion_map.shape == (target_height, target_width):
        return motion_map.astype(np.float32, copy=False)

    cv2 = _import_cv2()
    if cv2 is not None:
        resized = cv2.resize(motion_map.astype(np.float32), (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    else:
        resized = np.asarray(
            Image.fromarray(np.clip(motion_map * 255.0, 0, 255).astype(np.uint8)).resize(
                (target_width, target_height),
                resample=PIL_BILINEAR,
            )
        ).astype(np.float32) / 255.0
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def _format_status_markdown(
    headline: str,
    guidance: str,
    metrics: Mapping[str, Any],
    action_probe: Mapping[str, Any] | None = None,
) -> str:
    action_line = ""
    if action_probe is not None:
        action_line = (
            f"`hypothesis`: {action_probe.get('display_label')}  "
            f"`confidence`: {action_probe.get('confidence')}\n\n"
        )
    return (
        f"### {headline}\n"
        f"{guidance}\n\n"
        f"{action_line}"
        f"`backend`: {metrics.get('backend')}  "
        f"`mode`: {metrics.get('analysis_mode')}  "
        f"`motion`: {metrics.get('motion_energy')}  "
        f"`active`: {metrics.get('active_pixels')}  "
        f"`shift_px`: {metrics.get('camera_shift_px')}"
    )


def _format_guidance_markdown(camera_task: str, research_note: str) -> str:
    return f"### What To Do\n{camera_task}\n\n### Why It Matters\n{research_note}"


def _append_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    events = list(state.get("events", []))
    events.append(event)
    state["events"] = events[-MAX_EVENT_HISTORY:]


def _action_collection_tip(label: str) -> str:
    tips = {
        "wave": "Repeat the wave 3-5 times with the hand and forearm fully visible, then record several short clips from slightly different body positions.",
        "arm_raise": "Raise one or both arms from rest to above shoulder height, pause briefly at the top, then lower them under control.",
        "side_step": "Step left or right while keeping your full body in frame and leaving a still pose before and after the step.",
        "squat_like": "Perform a controlled squat or sit-to-stand with your hips and knees visible, and avoid camera movement during the rep.",
        "jumping_like": "Jump in place with enough room above your head and leave a still pose at the start and end of each repetition.",
    }
    return tips.get(
        label,
        "Keep one action per take, begin from stillness, perform the motion cleanly for 2-3 seconds, then return to a stable pose.",
    )


def _maybe_log_event(state: Mapping[str, Any], event: Mapping[str, Any]) -> None:
    frame_index = int(event.get("frame_index", 0))
    status = str(event.get("status", ""))
    should_log = frame_index <= 3 or frame_index % 15 == 0 or status != state.get("last_status")
    if should_log:
        logger.info(
            "Live frame analyzed | frame=%s | status=%s | action=%s | mode=%s | motion=%.4f | active=%.4f | brightness=%s | processing_ms=%s",
            frame_index,
            status,
            event.get("action_hypothesis"),
            event.get("analysis_mode"),
            float(event.get("motion_energy", 0.0)),
            float(event.get("active_pixels", 0.0)),
            event.get("brightness"),
            event.get("processing_ms"),
        )
