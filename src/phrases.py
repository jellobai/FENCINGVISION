from __future__ import annotations

import pandas as pd


def segment_phrases(frame_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the bout whenever tempo has been calm for a while after a burst."""
    features = frame_features.copy()
    features["calm"] = features["tempo_score"] < features["tempo_score"].median() * 0.7
    features["new_phrase_flag"] = (
        features["calm"].rolling(20, min_periods=1).sum().shift(fill_value=0) > 15
    ) & features["tempo_burst"]
    features["phrase_id"] = features["new_phrase_flag"].cumsum() + 1

    grouped = features.groupby("phrase_id", as_index=False).agg(
        start_time=("time_seconds", "min"),
        end_time=("time_seconds", "max"),
        start_distance=("distance_pixels", "first"),
        end_distance=("distance_pixels", "last"),
        left_forward_frames=("left_forward", "sum"),
        right_forward_frames=("right_forward", "sum"),
        left_retreat_frames=("left_retreat", "sum"),
        right_retreat_frames=("right_retreat", "sum"),
        tempo_bursts=("tempo_burst", "sum"),
    )
    grouped["duration"] = grouped["end_time"] - grouped["start_time"]
    return features, grouped
