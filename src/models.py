from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


@dataclass(slots=True)
class OverviewMetrics:
    total_phrases: int
    total_tempo_bursts: int
    near_simultaneous_phrases: int
    delayed_commitment_flags: int


@dataclass(slots=True)
class AnalysisResult:
    metadata: VideoMetadata
    frame_features: pd.DataFrame
    phrases: pd.DataFrame
    overview: OverviewMetrics
    detection_source: str


@dataclass(slots=True)
class PracticeFocus:
    title: str
    reason: str
    practice_idea: str


@dataclass(slots=True)
class TrainingReport:
    summary: str
    practice_focus: list[PracticeFocus]
