from __future__ import annotations

from pathlib import Path

from src.detector import detect_fencers
from src.features import compute_frame_features
from src.models import AnalysisResult, OverviewMetrics
from src.phrases import segment_phrases
from src.saber_rules import classify_phrase_events
from src.tracker import track_fencers
from src.video_io import load_video_metadata


def analyze_video(video_path: Path, athlete_side: str) -> AnalysisResult:
    metadata = load_video_metadata(video_path)
    detections, detection_source = detect_fencers(video_path=video_path, metadata=metadata)
    tracks = track_fencers(detections)
    frame_features = compute_frame_features(tracks=tracks, metadata=metadata)
    frame_features, phrases = segment_phrases(frame_features)
    phrases = classify_phrase_events(frame_features=frame_features, phrases=phrases)
    phrases["athlete_side"] = athlete_side
    phrases["athlete_moved_first"] = phrases["moved_first"] == athlete_side
    phrases["athlete_committed_first"] = phrases["committed_first"] == athlete_side

    overview = OverviewMetrics(
        total_phrases=int(len(phrases)),
        total_tempo_bursts=int(frame_features["tempo_burst"].sum()),
        near_simultaneous_phrases=int(phrases["near_simultaneous"].sum()),
        delayed_commitment_flags=int(phrases["delayed_commitment"].sum()),
    )
    return AnalysisResult(
        metadata=metadata,
        frame_features=frame_features,
        phrases=phrases,
        overview=overview,
        detection_source=detection_source,
    )
