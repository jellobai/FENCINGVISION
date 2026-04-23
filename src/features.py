from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import VideoMetadata


def compute_frame_features(tracks: pd.DataFrame, metadata: VideoMetadata) -> pd.DataFrame:
    features = tracks.copy()
    dt = 1.0 / metadata.fps

    features["distance_pixels"] = features["right_center_x"] - features["left_center_x"]
    features["tip_distance_pixels"] = features["right_tip_x"] - features["left_tip_x"]
    features["left_velocity"] = np.gradient(features["left_center_x"], dt)
    features["right_velocity"] = np.gradient(features["right_center_x"], dt)
    features["left_tip_speed"] = np.hypot(
        np.gradient(features["left_tip_x"], dt),
        np.gradient(features["left_tip_y"], dt),
    )
    features["right_tip_speed"] = np.hypot(
        np.gradient(features["right_tip_x"], dt),
        np.gradient(features["right_tip_y"], dt),
    )
    features["closing_speed"] = -np.gradient(features["distance_pixels"], dt)
    features["tempo_score"] = np.abs(features["closing_speed"]).rolling(9, min_periods=1).mean()

    features["left_forward"] = features["left_velocity"] > 25
    features["right_forward"] = features["right_velocity"] < -25
    features["left_retreat"] = features["left_velocity"] < -25
    features["right_retreat"] = features["right_velocity"] > 25

    distance_baseline = features["distance_pixels"].median()
    features["distance_state"] = np.select(
        [
            features["distance_pixels"] > distance_baseline * 1.08,
            features["distance_pixels"] < distance_baseline * 0.92,
        ],
        ["long", "close"],
        default="medium",
    )

    features["tempo_burst"] = features["tempo_score"] > (
        features["tempo_score"].rolling(45, min_periods=1).mean() * 1.25
    )

    features["pressure_owner"] = np.select(
        [
            features["left_forward"] & features["right_retreat"],
            features["right_forward"] & features["left_retreat"],
        ],
        ["left", "right"],
        default="neutral",
    )
    features["tip_pressure_owner"] = np.select(
        [
            features["left_tip_speed"] > features["right_tip_speed"] * 1.12,
            features["right_tip_speed"] > features["left_tip_speed"] * 1.12,
        ],
        ["left", "right"],
        default="neutral",
    )

    return features
