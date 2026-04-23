from __future__ import annotations

import numpy as np
import pandas as pd


def track_fencers(detections: pd.DataFrame) -> pd.DataFrame:
    tracked = detections.copy()
    tracked = tracked.sort_values("frame").reset_index(drop=True)
    tracked = _resolve_crossings(tracked)
    tracked["left_center_x"] = _smooth_series(tracked["left_center_x"])
    tracked["right_center_x"] = _smooth_series(tracked["right_center_x"])
    tracked["left_track_id"] = 1
    tracked["right_track_id"] = 2
    return tracked


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
