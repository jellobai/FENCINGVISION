from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models import VideoMetadata


def track_fencers(
    detections: pd.DataFrame,
    video_path: Path | None = None,
    metadata: VideoMetadata | None = None,
) -> pd.DataFrame:
    tracked = detections.copy()
    tracked = tracked.sort_values("frame").reset_index(drop=True)
    tracked = _resolve_crossings(tracked)
    tracked["left_center_x"] = _smooth_series(tracked["left_center_x"])
    tracked["right_center_x"] = _smooth_series(tracked["right_center_x"])
    tracked["left_track_id"] = 1
    tracked["right_track_id"] = 2

    if video_path is not None and metadata is not None:
        tracked = _track_saber_tips(
            tracked=tracked,
            video_path=video_path,
            metadata=metadata,
        )
    else:
        tracked = _attach_tip_fallbacks(tracked)

    return tracked


def _track_saber_tips(tracked: pd.DataFrame, video_path: Path, metadata: VideoMetadata) -> pd.DataFrame:
    try:
        import cv2  # type: ignore
    except ImportError:
        return _attach_tip_fallbacks(tracked)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return _attach_tip_fallbacks(tracked)

    tips_by_frame: dict[int, dict[str, float]] = {}
    previous_left_tip: tuple[float, float] | None = None
    previous_right_tip: tuple[float, float] | None = None
    tracked_by_frame = tracked.set_index("frame", drop=False)
    pose_estimator = _load_pose_estimator()

    for frame_number in range(metadata.frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_number not in tracked_by_frame.index:
            continue

        row = tracked_by_frame.loc[frame_number]
        left_pose = _estimate_arm_anchor(frame=frame, row=row, side="left", pose_estimator=pose_estimator)
        right_pose = _estimate_arm_anchor(frame=frame, row=row, side="right", pose_estimator=pose_estimator)
        left_tip = _detect_tip_for_side(frame, row=row, side="left", previous_tip=previous_left_tip, arm_anchor=left_pose)
        right_tip = _detect_tip_for_side(frame, row=row, side="right", previous_tip=previous_right_tip, arm_anchor=right_pose)

        if left_tip is None:
            left_tip = _fallback_tip_from_box(row=row, side="left")
        if right_tip is None:
            right_tip = _fallback_tip_from_box(row=row, side="right")

        previous_left_tip = left_tip
        previous_right_tip = right_tip
        tips_by_frame[int(frame_number)] = {
            "left_tip_x": float(left_tip[0]),
            "left_tip_y": float(left_tip[1]),
            "right_tip_x": float(right_tip[0]),
            "right_tip_y": float(right_tip[1]),
            "left_wrist_x": float(left_pose["wrist"][0]) if left_pose is not None else np.nan,
            "left_wrist_y": float(left_pose["wrist"][1]) if left_pose is not None else np.nan,
            "right_wrist_x": float(right_pose["wrist"][0]) if right_pose is not None else np.nan,
            "right_wrist_y": float(right_pose["wrist"][1]) if right_pose is not None else np.nan,
        }

    capture.release()

    if not tips_by_frame:
        return _attach_tip_fallbacks(tracked)

    tips_frame = pd.DataFrame.from_dict(tips_by_frame, orient="index").reset_index(names="frame")
    merged = tracked.merge(tips_frame, on="frame", how="left")
    return _smooth_tip_columns(_fill_tip_gaps(merged))


def _detect_tip_for_side(
    frame: np.ndarray,
    row: pd.Series,
    side: str,
    previous_tip: tuple[float, float] | None,
    arm_anchor: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float] | None:
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover
        return None

    center_x = float(row[f"{side}_center_x"])
    center_y = float(row.get(f"{side}_center_y", frame.shape[0] * 0.55))
    box_width = float(row.get(f"{side}_width", 90.0))
    box_height = float(row.get(f"{side}_height", frame.shape[0] * 0.34))

    wrist_x, wrist_y = _weapon_wrist(row=row, side=side, arm_anchor=arm_anchor)
    elbow_x, elbow_y = _weapon_elbow(row=row, side=side, arm_anchor=arm_anchor)
    forearm_dx = wrist_x - elbow_x
    forearm_dy = wrist_y - elbow_y
    forearm_length = max(float(np.hypot(forearm_dx, forearm_dy)), box_width * 0.35)
    direction_x = forearm_dx / forearm_length if forearm_length else (1.0 if side == "left" else -1.0)
    direction_y = forearm_dy / forearm_length if forearm_length else -0.1

    horizontal_reach = max(box_width * 1.6, forearm_length * 2.4)
    vertical_reach = max(box_height * 0.65, forearm_length * 0.9)
    if side == "left":
        x1 = max(int(min(wrist_x, center_x) - box_width * 0.2), 0)
        x2 = min(int(wrist_x + horizontal_reach), frame.shape[1] - 1)
    else:
        x1 = max(int(wrist_x - horizontal_reach), 0)
        x2 = min(int(max(wrist_x, center_x) + box_width * 0.2), frame.shape[1] - 1)

    y1 = max(int(min(wrist_y, center_y) - vertical_reach), 0)
    y2 = min(int(max(wrist_y, center_y) + vertical_reach * 0.45), frame.shape[0] - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 70, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=24,
        minLineLength=max(int(box_width * 0.45), 18),
        maxLineGap=12,
    )
    if lines is None:
        return _tip_from_edges(edges=edges, x_offset=x1, y_offset=y1, side=side, previous_tip=previous_tip)

    best_tip: tuple[float, float] | None = None
    best_score: float | None = None

    for line in lines[:, 0]:
        x_start, y_start, x_end, y_end = [int(value) for value in line]
        for candidate_x, candidate_y in ((x_start, y_start), (x_end, y_end)):
            absolute_x = candidate_x + x1
            absolute_y = candidate_y + y1
            direction_ok = absolute_x > wrist_x if side == "left" else absolute_x < wrist_x
            if not direction_ok:
                continue

            distance = float(np.hypot(absolute_x - wrist_x, absolute_y - wrist_y))
            vertical_penalty = abs(absolute_y - wrist_y) * 0.35
            direction_score = ((absolute_x - wrist_x) * direction_x) + ((absolute_y - wrist_y) * direction_y)
            continuity_penalty = 0.0
            if previous_tip is not None:
                continuity_penalty = np.hypot(absolute_x - previous_tip[0], absolute_y - previous_tip[1]) * 0.18
            score = distance + direction_score * 0.4 - vertical_penalty - continuity_penalty
            if best_score is None or score > best_score:
                best_score = score
                best_tip = (float(absolute_x), float(absolute_y))

    if best_tip is not None:
        return best_tip
    return _tip_from_edges(edges=edges, x_offset=x1, y_offset=y1, side=side, previous_tip=previous_tip)


def _tip_from_edges(
    edges: np.ndarray,
    x_offset: int,
    y_offset: int,
    side: str,
    previous_tip: tuple[float, float] | None,
) -> tuple[float, float] | None:
    active_points = np.argwhere(edges > 0)
    if len(active_points) == 0:
        return None

    if side == "left":
        order = np.lexsort((active_points[:, 0], active_points[:, 1]))
    else:
        order = np.lexsort((active_points[:, 0], -active_points[:, 1]))

    best = active_points[order[-1]]
    candidate = (float(best[1] + x_offset), float(best[0] + y_offset))
    if previous_tip is None:
        return candidate

    if np.hypot(candidate[0] - previous_tip[0], candidate[1] - previous_tip[1]) > 180:
        return previous_tip
    return candidate


def _fallback_tip_from_box(row: pd.Series, side: str) -> tuple[float, float]:
    center_x = float(row[f"{side}_center_x"])
    center_y = float(row.get(f"{side}_center_y", 0.0))
    box_width = float(row.get(f"{side}_width", 90.0))
    box_height = float(row.get(f"{side}_height", 120.0))
    if side == "left":
        return center_x + box_width * 0.8, center_y - box_height * 0.2
    return center_x - box_width * 0.8, center_y - box_height * 0.2


def _attach_tip_fallbacks(tracked: pd.DataFrame) -> pd.DataFrame:
    tracked = tracked.copy()
    tracked["left_tip_x"] = tracked["left_center_x"] + tracked["left_width"] * 0.8
    tracked["left_tip_y"] = tracked.get("left_center_y", tracked["left_center_x"] * 0 + 0) - tracked.get(
        "left_height",
        tracked["left_width"] * 1.3,
    ) * 0.2
    tracked["right_tip_x"] = tracked["right_center_x"] - tracked["right_width"] * 0.8
    tracked["right_tip_y"] = tracked.get("right_center_y", tracked["right_center_x"] * 0 + 0) - tracked.get(
        "right_height",
        tracked["right_width"] * 1.3,
    ) * 0.2
    tracked["left_wrist_x"] = tracked["left_center_x"] + tracked["left_width"] * 0.25
    tracked["left_wrist_y"] = tracked.get("left_center_y", tracked["left_center_x"] * 0 + 0) - tracked.get(
        "left_height",
        tracked["left_width"] * 1.3,
    ) * 0.18
    tracked["right_wrist_x"] = tracked["right_center_x"] - tracked["right_width"] * 0.25
    tracked["right_wrist_y"] = tracked.get("right_center_y", tracked["right_center_x"] * 0 + 0) - tracked.get(
        "right_height",
        tracked["right_width"] * 1.3,
    ) * 0.18
    return tracked


def _fill_tip_gaps(tracked: pd.DataFrame) -> pd.DataFrame:
    filled = tracked.copy()
    for column in [
        "left_tip_x",
        "left_tip_y",
        "right_tip_x",
        "right_tip_y",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
    ]:
        filled[column] = filled[column].interpolate(limit_direction="both")
    return filled


def _smooth_tip_columns(tracked: pd.DataFrame) -> pd.DataFrame:
    smoothed = tracked.copy()
    for column in [
        "left_tip_x",
        "left_tip_y",
        "right_tip_x",
        "right_tip_y",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
    ]:
        smoothed[column] = _smooth_series(smoothed[column], window=7)
    return smoothed


def _resolve_crossings(tracked: pd.DataFrame) -> pd.DataFrame:
    left = tracked["left_center_x"].to_numpy(dtype=float)
    right = tracked["right_center_x"].to_numpy(dtype=float)
    swap_mask = left >= right
    if not np.any(swap_mask):
        return tracked

    columns = [
        ("left_center_x", "right_center_x"),
        ("left_width", "right_width"),
    ]
    optional_columns = [
        ("left_center_y", "right_center_y"),
        ("left_height", "right_height"),
        ("left_tip_x", "right_tip_x"),
        ("left_tip_y", "right_tip_y"),
    ]
    for pair in optional_columns:
        if pair[0] in tracked.columns and pair[1] in tracked.columns:
            columns.append(pair)

    for left_column, right_column in columns:
        left_values = tracked[left_column].to_numpy(copy=True)
        right_values = tracked[right_column].to_numpy(copy=True)
        tracked.loc[swap_mask, left_column] = right_values[swap_mask]
        tracked.loc[swap_mask, right_column] = left_values[swap_mask]

    tracked["right_center_x"] = np.maximum(
        tracked["right_center_x"].to_numpy(dtype=float),
        tracked["left_center_x"].to_numpy(dtype=float) + 1.0,
    )
    return tracked


def _smooth_series(values: pd.Series, window: int = 9) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    if len(array) <= 2:
        return array
    effective_window = min(window, len(array))
    if effective_window % 2 == 0:
        effective_window = max(1, effective_window - 1)
    kernel = np.ones(effective_window, dtype=float) / effective_window
    return np.convolve(array, kernel, mode="same")


def _load_pose_estimator():
    try:
        import mediapipe as mp  # type: ignore
    except ImportError:
        return None
    try:
        return mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
    except Exception:
        return None


def _estimate_arm_anchor(
    frame: np.ndarray,
    row: pd.Series,
    side: str,
    pose_estimator,
) -> dict[str, tuple[float, float]] | None:
    if pose_estimator is None:
        return None

    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError:  # pragma: no cover
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_estimator.process(rgb)
    landmarks = getattr(result, "pose_landmarks", None)
    if landmarks is None:
        return None

    mp_pose = mp.solutions.pose
    index_map = {
        "left": (
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value,
        ),
        "right": (
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value,
        ),
    }
    elbow_idx, wrist_idx = index_map[side]
    elbow = landmarks.landmark[elbow_idx]
    wrist = landmarks.landmark[wrist_idx]
    if elbow.visibility < 0.25 or wrist.visibility < 0.25:
        return None

    elbow_xy = (float(elbow.x * frame.shape[1]), float(elbow.y * frame.shape[0]))
    wrist_xy = (float(wrist.x * frame.shape[1]), float(wrist.y * frame.shape[0]))
    box_center_x = float(row[f"{side}_center_x"])
    box_center_y = float(row.get(f"{side}_center_y", frame.shape[0] * 0.55))
    box_width = float(row.get(f"{side}_width", 90.0))
    box_height = float(row.get(f"{side}_height", frame.shape[0] * 0.34))
    if abs(wrist_xy[0] - box_center_x) > box_width * 1.8 or abs(wrist_xy[1] - box_center_y) > box_height * 1.6:
        return None

    return {"elbow": elbow_xy, "wrist": wrist_xy}


def _weapon_wrist(
    row: pd.Series,
    side: str,
    arm_anchor: dict[str, tuple[float, float]] | None,
) -> tuple[float, float]:
    if arm_anchor is not None:
        return arm_anchor["wrist"]
    return (
        float(row.get(f"{side}_wrist_x", row[f"{side}_center_x"])),
        float(row.get(f"{side}_wrist_y", row.get(f"{side}_center_y", 0.0))),
    )


def _weapon_elbow(
    row: pd.Series,
    side: str,
    arm_anchor: dict[str, tuple[float, float]] | None,
) -> tuple[float, float]:
    if arm_anchor is not None:
        return arm_anchor["elbow"]
    center_x = float(row[f"{side}_center_x"])
    center_y = float(row.get(f"{side}_center_y", 0.0))
    width = float(row.get(f"{side}_width", 90.0))
    height = float(row.get(f"{side}_height", 120.0))
    offset = width * 0.12 if side == "left" else -width * 0.12
    return center_x + offset, center_y - height * 0.08
