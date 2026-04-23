from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from src.models import AnalysisResult, PracticeFocus, TrainingReport


def build_training_report(analysis: AnalysisResult) -> TrainingReport:
    phrases = analysis.phrases
    athlete = str(phrases["athlete_side"].iloc[0]) if len(phrases) else "selected"

    launch_rate = float(phrases["athlete_moved_first"].mean()) if len(phrases) else 0.0
    delayed_rate = float(phrases["delayed_commitment"].mean()) if len(phrases) else 0.0
    near_sim_rate = float(phrases["near_simultaneous"].mean()) if len(phrases) else 0.0

    summary = (
        f"{analysis.overview.total_phrases} phrases detected. "
        f"The {athlete} side launched first in {launch_rate:.0%} of phrases, "
        f"with delayed commitment flagged in {delayed_rate:.0%}. "
        f"Near-simultaneous openings appeared in {near_sim_rate:.0%} of phrases."
    )

    practice_focus = [
        PracticeFocus(
            title="Sharpen first commitment",
            reason="Delayed commitment appeared when first movement did not convert into a tempo-backed burst.",
            practice_idea="Use short start drills that require a committed second acceleration after the opening step.",
        ),
        PracticeFocus(
            title="Control pressure transitions",
            reason="Pressure ownership changed across phrases instead of staying stable through the exchange.",
            practice_idea="Run phrase drills where you hold forward pressure for two movement phases before finishing.",
        ),
        PracticeFocus(
            title="Own the opening rhythm",
            reason="Tempo bursts clustered around phrase starts, making the first 0.5 seconds decisive.",
            practice_idea="Practice first-step timing off the line with one explosive decision and no reset pause.",
        ),
    ]
    return TrainingReport(summary=summary, practice_focus=practice_focus)


def render_phrase_table(phrases: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "phrase_id",
        "start_time",
        "end_time",
        "moved_first",
        "committed_first",
        "pressure_owner",
        "distance_trend",
        "tempo_label",
        "strip_label",
        "near_simultaneous",
        "delayed_commitment",
    ]
    table = phrases.loc[:, columns].copy()
    table["start_time"] = table["start_time"].map(lambda value: f"{value:.2f}s")
    table["end_time"] = table["end_time"].map(lambda value: f"{value:.2f}s")
    return table


def build_report_payload(analysis: AnalysisResult) -> dict:
    report = build_training_report(analysis)
    phrase_rows = render_phrase_table(analysis.phrases).to_dict(orient="records")

    key_metrics = {
        "phrases_detected": analysis.overview.total_phrases,
        "tempo_bursts": analysis.overview.total_tempo_bursts,
        "near_simultaneous_phrases": analysis.overview.near_simultaneous_phrases,
        "delayed_commitment_flags": analysis.overview.delayed_commitment_flags,
    }

    distance_series = analysis.frame_features.loc[:, ["time_seconds", "distance_pixels", "tempo_burst"]]
    strip_series = analysis.frame_features.loc[:, ["time_seconds", "left_center_x", "right_center_x"]]

    return {
        "summary": report.summary,
        "practice_focus": [asdict(item) for item in report.practice_focus],
        "metrics": key_metrics,
        "analysis_meta": {
            "detection_source": analysis.detection_source,
            "fps": round(analysis.metadata.fps, 2),
            "duration_seconds": round(analysis.metadata.duration_seconds, 2),
            "frame_count": analysis.metadata.frame_count,
            "resolution": f"{analysis.metadata.width}x{analysis.metadata.height}",
            "tracking_mode": "left-right continuity smoothing",
        },
        "phrases": phrase_rows,
        "charts": {
            "distance": distance_series.to_dict(orient="records"),
            "strip": strip_series.to_dict(orient="records"),
        },
    }


def attach_artifact_metadata(payload: dict, annotated_video_url: str | None) -> dict:
    payload["artifacts"] = {
        "annotated_video_url": annotated_video_url,
        "annotated_video_available": bool(annotated_video_url),
    }
    return payload
