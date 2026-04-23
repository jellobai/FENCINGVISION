from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import VideoMetadata


def detect_fencers(video_path: Path, metadata: VideoMetadata) -> tuple[pd.DataFrame, str]:
    """Detect left/right fencers from foreground motion, falling back to synthetic scaffolding."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return _synthetic_detections(metadata), "synthetic fallback"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return _synthetic_detections(metadata), "synthetic fallback"

    subtractor = cv2.createBackgroundSubtractorMOG2(history=240, varThreshold=36, detectShadows=False)
    detections: list[dict[str, float]] = []
    previous_left_box: tuple[int, int, int, int] | None = None
    previous_right_box: tuple[int, int, int, int] | None = None

    for frame_index in range(metadata.frame_count):
        ok, frame = capture.read()
        if not ok:
            break

        strip_frame, x_offset, y_offset = _extract_strip_roi(frame)
        if strip_frame.size == 0:
            continue

        mask = subtractor.apply(strip_frame)
        person_boxes = _find_candidate_boxes(mask=mask, frame_shape=strip_frame.shape)
        if not person_boxes:
            continue

        left_box, right_box = _assign_fencers(
            boxes=person_boxes,
            frame_width=strip_frame.shape[1],
            previous_left_box=previous_left_box,
            previous_right_box=previous_right_box,
        )
        if left_box is None or right_box is None:
            continue

        previous_left_box = left_box
        previous_right_box = right_box

        detections.append(
            {
                "frame": float(frame_index),
                "time_seconds": frame_index / metadata.fps,
                "left_center_x": float(left_box[0] + left_box[2] / 2 + x_offset),
                "right_center_x": float(right_box[0] + right_box[2] / 2 + x_offset),
                "left_width": float(left_box[2]),
                "right_width": float(right_box[2]),
                "left_center_y": float(left_box[1] + left_box[3] / 2 + y_offset),
                "right_center_y": float(right_box[1] + right_box[3] / 2 + y_offset),
                "left_height": float(left_box[3]),
                "right_height": float(right_box[3]),
            }
        )

    capture.release()

    sampled = pd.DataFrame(detections)
    if len(sampled) < max(8, int(metadata.fps // 2)):
        return _synthetic_detections(metadata), "synthetic fallback"

    interpolated = _interpolate_detections(sampled=sampled, metadata=metadata)
    if interpolated is None:
        return _synthetic_detections(metadata), "synthetic fallback"

    return interpolated, "foreground contour detector"


def _extract_strip_roi(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    height, width = frame.shape[:2]
    top = int(height * 0.16)
    bottom = int(height * 0.9)
    left = int(width * 0.04)
    right = int(width * 0.96)
    return frame[top:bottom, left:right], left, top


def _find_candidate_boxes(mask: np.ndarray, frame_shape: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover
        return []

    kernel = np.ones((5, 5), dtype=np.uint8)
    refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_height, frame_width = frame_shape[:2]
    min_area = max(220, int(frame_height * frame_width * 0.003))
    boxes: list[tuple[int, int, int, int]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / max(w, 1)
        if h < frame_height * 0.14 or w < frame_width * 0.03:
            continue
        if aspect_ratio < 0.9:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    return boxes[:6]


def _assign_fencers(
    boxes: list[tuple[int, int, int, int]],
    frame_width: int,
    previous_left_box: tuple[int, int, int, int] | None,
    previous_right_box: tuple[int, int, int, int] | None,
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    if len(boxes) == 1:
        single = boxes[0]
        center_x = single[0] + single[2] / 2
        if center_x < frame_width / 2:
            return single, previous_right_box
        return previous_left_box, single

    best_pair: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None = None
    best_score: float | None = None

    for index, left_candidate in enumerate(boxes):
        for right_candidate in boxes[index + 1 :]:
            ordered_left, ordered_right = sorted(
                [left_candidate, right_candidate],
                key=lambda box: box[0] + box[2] / 2,
            )
            score = _pair_score(
                left_box=ordered_left,
                right_box=ordered_right,
                frame_width=frame_width,
                previous_left_box=previous_left_box,
                previous_right_box=previous_right_box,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (ordered_left, ordered_right)

    if best_pair is None:
        return None, None
    return best_pair


def _pair_score(
    left_box: tuple[int, int, int, int],
    right_box: tuple[int, int, int, int],
    frame_width: int,
    previous_left_box: tuple[int, int, int, int] | None,
    previous_right_box: tuple[int, int, int, int] | None,
) -> float:
    left_center = left_box[0] + left_box[2] / 2
    right_center = right_box[0] + right_box[2] / 2
    if left_center >= right_center:
        return -1e9

    separation = right_center - left_center
    lane_bias = (frame_width - abs(left_center - frame_width * 0.25) - abs(right_center - frame_width * 0.75)) / frame_width
    area_score = (left_box[2] * left_box[3] + right_box[2] * right_box[3]) / max(frame_width, 1)
    score = separation * 0.02 + lane_bias * 200 + area_score

    if previous_left_box is not None:
        score -= abs(left_center - (previous_left_box[0] + previous_left_box[2] / 2)) * 0.8
    if previous_right_box is not None:
        score -= abs(right_center - (previous_right_box[0] + previous_right_box[2] / 2)) * 0.8

    return score


def _interpolate_detections(sampled: pd.DataFrame, metadata: VideoMetadata) -> pd.DataFrame | None:
    frame_index = np.arange(metadata.frame_count)
    if len(frame_index) < 2:
        return None

    sampled = sampled.sort_values("frame").drop_duplicates("frame")
    sampled_frames = sampled["frame"].to_numpy(dtype=float)
    if len(sampled_frames) < 2:
        return None

    interpolated = pd.DataFrame(
        {
            "frame": frame_index,
            "time_seconds": frame_index / metadata.fps,
        }
    )

    for column in [
        "left_center_x",
        "right_center_x",
        "left_width",
        "right_width",
        "left_center_y",
        "right_center_y",
        "left_height",
        "right_height",
    ]:
        interpolated[column] = np.interp(frame_index, sampled_frames, sampled[column].to_numpy(dtype=float))

    interpolated["left_center_x"] = _smooth_track(interpolated["left_center_x"].to_numpy())
    interpolated["right_center_x"] = _smooth_track(interpolated["right_center_x"].to_numpy())
    interpolated["left_center_y"] = _smooth_track(interpolated["left_center_y"].to_numpy())
    interpolated["right_center_y"] = _smooth_track(interpolated["right_center_y"].to_numpy())

    minimum_gap = metadata.width * 0.05
    interpolated["right_center_x"] = np.maximum(
        interpolated["right_center_x"].to_numpy(),
        interpolated["left_center_x"].to_numpy() + minimum_gap,
    )

    return interpolated


def _smooth_track(values: np.ndarray, window: int = 11) -> np.ndarray:
    if len(values) <= 2:
        return values
    effective_window = min(window, len(values))
    if effective_window % 2 == 0:
        effective_window = max(1, effective_window - 1)
    kernel = np.ones(effective_window, dtype=float) / effective_window
    return np.convolve(values, kernel, mode="same")


def _synthetic_detections(metadata: VideoMetadata) -> pd.DataFrame:
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
            "left_center_y": metadata.height * 0.55,
            "right_center_y": metadata.height * 0.55,
            "left_height": metadata.height * 0.34,
            "right_height": metadata.height * 0.34,
        }
    )
