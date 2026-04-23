from __future__ import annotations

import pandas as pd


def classify_phrase_events(frame_features: pd.DataFrame, phrases: pd.DataFrame) -> pd.DataFrame:
    phrase_rows: list[dict[str, object]] = []

    for phrase in phrases.itertuples(index=False):
        segment = frame_features[frame_features["phrase_id"] == phrase.phrase_id]
        first_left_forward = _first_time(segment, "left_forward")
        first_right_forward = _first_time(segment, "right_forward")
        first_left_burst = _first_time(segment, "left_forward", require_tempo_burst=True)
        first_right_burst = _first_time(segment, "right_forward", require_tempo_burst=True)

        moved_first = _decide_first(first_left_forward, first_right_forward)
        committed_first = _decide_first(first_left_burst, first_right_burst)
        near_simultaneous = _is_near_simultaneous(first_left_burst, first_right_burst)

        pressure_owner = segment["pressure_owner"].value_counts().idxmax()
        delayed_commitment = committed_first != "tie" and moved_first != "tie" and committed_first != moved_first

        phrase_rows.append(
            {
                "phrase_id": phrase.phrase_id,
                "moved_first": moved_first,
                "committed_first": committed_first,
                "pressure_owner": pressure_owner,
                "near_simultaneous": near_simultaneous,
                "delayed_commitment": delayed_commitment,
                "distance_trend": _distance_trend(segment),
                "tempo_label": _tempo_label(segment),
                "strip_label": _strip_label(segment),
            }
        )

    return phrases.merge(pd.DataFrame(phrase_rows), on="phrase_id", how="left")


def _first_time(segment: pd.DataFrame, column: str, require_tempo_burst: bool = False) -> float | None:
    mask = segment[column]
    if require_tempo_burst:
        mask = mask & segment["tempo_burst"]
    subset = segment.loc[mask, "time_seconds"]
    if subset.empty:
        return None
    return float(subset.iloc[0])


def _decide_first(left_time: float | None, right_time: float | None) -> str:
    if left_time is None and right_time is None:
        return "unknown"
    if left_time is None:
        return "right"
    if right_time is None:
        return "left"
    if abs(left_time - right_time) <= 0.12:
        return "tie"
    return "left" if left_time < right_time else "right"


def _is_near_simultaneous(left_time: float | None, right_time: float | None) -> bool:
    if left_time is None or right_time is None:
        return False
    return abs(left_time - right_time) <= 0.18


def _distance_trend(segment: pd.DataFrame) -> str:
    delta = float(segment["distance_pixels"].iloc[-1] - segment["distance_pixels"].iloc[0])
    if delta < -40:
        return "distance closed"
    if delta > 40:
        return "distance expanded"
    return "distance steady"


def _tempo_label(segment: pd.DataFrame) -> str:
    burst_count = int(segment["tempo_burst"].sum())
    if burst_count >= 20:
        return "high burst"
    if burst_count >= 8:
        return "moderate burst"
    return "calm"


def _strip_label(segment: pd.DataFrame) -> str:
    left_mean = float(segment["left_center_x"].mean())
    right_mean = float(segment["right_center_x"].mean())
    center = (left_mean + right_mean) / 2
    if center < segment["left_center_x"].min() + 220:
        return "left side"
    if center > segment["right_center_x"].max() - 220:
        return "right side"
    return "middle"
