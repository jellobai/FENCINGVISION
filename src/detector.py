from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import VideoMetadata


def detect_fencers(video_path: Path, metadata: VideoMetadata) -> tuple[pd.DataFrame, str]:
    """Estimate left/right fencer strip positions from video motion when possible."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return _synthetic_detections(metadata), "synthetic fallback"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return _synthetic_detections(metadata), "synthetic fallback"

    sample_stride = max(int(round(metadata.fps / 6.0)), 1)
    sampled_rows: list[dict[str, float]] = []
    previous_band: np.ndarray | None = None

    while True:
        frame_id = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = capture.read()
        if not ok:
            break
        if frame_id % sample_stride != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        band = _extract_strip_band(gray)
        if band.size == 0:
            previous_band = None
            continue

        if previous_band is None:
            previous_band = band
            continue

        motion_mask = _build_motion_mask(current=band, previous=previous_band)
        previous_band = band

        left_center, left_width = _lane_estimate(motion_mask, 0.0, 0.5)
        right_center, right_width = _lane_estimate(motion_mask, 0.5, 1.0)
        if left_center is None or right_center is None:
            continue

        sampled_rows.append(
            {
                "frame": float(frame_id),
                "left_center_x": float(left_center),
                "right_center_x": float(right_center),
                "left_width": float(left_width),
                "right_width": float(right_width),
            }
        )

    capture.release()

    sampled = pd.DataFrame(sampled_rows)
    if len(sampled) < 4:
        return _synthetic_detections(metadata), "synthetic fallback"

    detections = _interpolate_detections(sampled=sampled, metadata=metadata)
    if detections is None:
        return _synthetic_detections(metadata), "synthetic fallback"
    return detections, "motion-derived baseline"


def _extract_strip_band(gray_frame: np.ndarray) -> np.ndarray:
    height = gray_frame.shape[0]
    top = int(height * 0.22)
    bottom = int(height * 0.82)
    band = gray_frame[top:bottom, :]
    return band


def _build_motion_mask(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    current_blur = _blur_frame(current)
    previous_blur = _blur_frame(previous)
    delta = np.abs(current_blur.astype(np.int16) - previous_blur.astype(np.int16)).astype(np.uint8)

    threshold = max(18, int(delta.mean() + delta.std()))
    mask = (delta > threshold).astype(np.uint8)
    return mask


def _blur_frame(frame: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover
        return frame
    return cv2.GaussianBlur(frame, (9, 9), 0)


def _lane_estimate(mask: np.ndarray, start_fraction: float, end_fraction: float) -> tuple[float | None, float]:
    width = mask.shape[1]
    start = int(width * start_fraction)
    end = int(width * end_fraction)
    if end <= start:
        return None, 0.0

    lane = mask[:, start:end]
    column_energy = lane.sum(axis=0).astype(float)
    if not np.any(column_energy):
        return None, 0.0

    column_energy = _smooth_series(column_energy, window=21)
    threshold = max(column_energy.mean() * 1.1, np.percentile(column_energy, 70))
    active = column_energy >= threshold
    if not np.any(active):
        active = column_energy >= column_energy.max() * 0.65
    if not np.any(active):
        return None, 0.0

    active_positions = np.flatnonzero(active)
    weights = column_energy[active]
    center = start + float(np.average(active_positions, weights=weights))
    spread = float(active_positions[-1] - active_positions[0] + 1)
    return center, max(spread, 48.0)


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) <= 2:
        return values
    effective_window = min(window, len(values))
    if effective_window % 2 == 0:
        effective_window = max(1, effective_window - 1)
    kernel = np.ones(effective_window, dtype=float) / effective_window
    return np.convolve(values, kernel, mode="same")


def _interpolate_detections(sampled: pd.DataFrame, metadata: VideoMetadata) -> pd.DataFrame | None:
    frame_index = np.arange(metadata.frame_count)
    if len(frame_index) == 0:
        return None

    sampled = sampled.sort_values("frame").drop_duplicates("frame")
    sampled_frames = sampled["frame"].to_numpy(dtype=float)
    if len(sampled_frames) < 2:
        return None

    left_center = np.interp(frame_index, sampled_frames, sampled["left_center_x"])
    right_center = np.interp(frame_index, sampled_frames, sampled["right_center_x"])
    left_width = np.interp(frame_index, sampled_frames, sampled["left_width"])
    right_width = np.interp(frame_index, sampled_frames, sampled["right_width"])

    min_gap = metadata.width * 0.08
    right_center = np.maximum(right_center, left_center + min_gap)

    detections = pd.DataFrame(
        {
            "frame": frame_index,
            "time_seconds": frame_index / metadata.fps,
            "left_center_x": left_center,
            "right_center_x": right_center,
            "left_width": left_width,
            "right_width": right_width,
        }
    )
    return detections


def _synthetic_detections(metadata: VideoMetadata) -> pd.DataFrame:
    """Deterministic placeholder detections used when video-derived tracking is unavailable."""
    frame_index = np.arange(metadata.frame_count)
    time_seconds = frame_index / metadata.fps

    left_center = metadata.width * 0.25 + 70 * np.sin(time_seconds * 1.2)
    right_center = metadata.width * 0.75 + 70 * np.sin(time_seconds * 1.1 + 1.6)

    burst_pattern = np.maximum(0.0, np.sin(time_seconds * 0.8)) ** 3
    left_center = left_center + 120 * burst_pattern
    right_center = right_center - 120 * burst_pattern

    return pd.DataFrame(
        {
            "frame": frame_index,
            "time_seconds": time_seconds,
            "left_center_x": left_center,
            "right_center_x": right_center,
            "left_width": 90.0,
            "right_width": 90.0,
        }
    )
