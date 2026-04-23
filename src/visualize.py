from __future__ import annotations

from pathlib import Path
import subprocess

import plotly.express as px
import pandas as pd

from src.models import AnalysisResult


def build_distance_chart(frame_features: pd.DataFrame):
    chart = px.line(
        frame_features,
        x="time_seconds",
        y="distance_pixels",
        title="Inter-fencer distance",
    )
    burst_points = frame_features[frame_features["tempo_burst"]]
    if not burst_points.empty:
        chart.add_scatter(
            x=burst_points["time_seconds"],
            y=burst_points["distance_pixels"],
            mode="markers",
            name="Tempo burst",
        )
    chart.update_layout(showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
    return chart


def build_strip_chart(frame_features: pd.DataFrame):
    plot_data = frame_features.melt(
        id_vars=["time_seconds"],
        value_vars=["left_center_x", "right_center_x"],
        var_name="fencer",
        value_name="strip_x",
    )
    chart = px.line(
        plot_data,
        x="time_seconds",
        y="strip_x",
        color="fencer",
        title="Estimated strip position",
    )
    chart.update_layout(showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
    return chart


def render_annotated_video(
    analysis: AnalysisResult,
    source_video_path: Path,
    output_video_path: Path,
) -> Path | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        return None

    fps = float(capture.get(cv2.CAP_PROP_FPS) or analysis.metadata.fps or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or analysis.metadata.width)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or analysis.metadata.height)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_video_path.with_name(f"{output_video_path.stem}_raw.mp4")
    writer = cv2.VideoWriter(
        str(intermediate_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 1 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        return None

    frame_features = analysis.frame_features.set_index("frame", drop=False)
    frame_limit = int(min(analysis.metadata.frame_count, len(frame_features)))
    athlete_side = str(analysis.phrases["athlete_side"].iloc[0]) if len(analysis.phrases) else "left"

    for frame_number in range(frame_limit):
        ok, frame = capture.read()
        if not ok:
            break

        if frame_number not in frame_features.index:
            writer.write(frame)
            continue

        row = frame_features.loc[frame_number]
        _draw_overlay(frame, row=row, athlete_side=athlete_side)
        writer.write(frame)

    writer.release()
    capture.release()
    if not intermediate_path.exists():
        return None

    transcoded_path = _transcode_for_browser(intermediate_path=intermediate_path, output_video_path=output_video_path)
    if transcoded_path is not None:
        intermediate_path.unlink(missing_ok=True)
        return transcoded_path

    intermediate_path.replace(output_video_path)
    return output_video_path if output_video_path.exists() else None


def _transcode_for_browser(intermediate_path: Path, output_video_path: Path) -> Path | None:
    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path is None:
        return None

    command = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(intermediate_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video_path),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return None

    if result.returncode != 0 or not output_video_path.exists():
        return None
    return output_video_path


def _find_ffmpeg() -> Path | None:
    for candidate in (
        "ffmpeg",
        "ffmpeg.exe",
        r"C:\Users\jbai01\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe",
    ):
        try:
            result = subprocess.run(
                [candidate, "-version"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        if result.returncode == 0:
            return Path(candidate)
    return None


def _draw_overlay(frame, row: pd.Series, athlete_side: str) -> None:
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover
        return

    colors = {
        "left": (36, 88, 230),
        "right": (60, 180, 75),
        "athlete": (20, 185, 220),
    }

    _draw_fencer_box(
        frame,
        center_x=float(row["left_center_x"]),
        center_y=float(row.get("left_center_y", frame.shape[0] * 0.55)),
        width=float(row.get("left_width", 90.0)),
        height=float(row.get("left_height", frame.shape[0] * 0.34)),
        label="LEFT",
        color=colors["athlete"] if athlete_side == "left" else colors["left"],
    )
    _draw_saber_tip(
        frame,
        hand_x=float(row.get("left_wrist_x", row["left_center_x"])),
        hand_y=float(row.get("left_wrist_y", row.get("left_center_y", frame.shape[0] * 0.55))),
        tip_x=float(row.get("left_tip_x", row["left_center_x"])),
        tip_y=float(row.get("left_tip_y", row.get("left_center_y", frame.shape[0] * 0.55))),
        color=colors["athlete"] if athlete_side == "left" else colors["left"],
    )
    _draw_fencer_box(
        frame,
        center_x=float(row["right_center_x"]),
        center_y=float(row.get("right_center_y", frame.shape[0] * 0.55)),
        width=float(row.get("right_width", 90.0)),
        height=float(row.get("right_height", frame.shape[0] * 0.34)),
        label="RIGHT",
        color=colors["athlete"] if athlete_side == "right" else colors["right"],
    )
    _draw_saber_tip(
        frame,
        hand_x=float(row.get("right_wrist_x", row["right_center_x"])),
        hand_y=float(row.get("right_wrist_y", row.get("right_center_y", frame.shape[0] * 0.55))),
        tip_x=float(row.get("right_tip_x", row["right_center_x"])),
        tip_y=float(row.get("right_tip_y", row.get("right_center_y", frame.shape[0] * 0.55))),
        color=colors["athlete"] if athlete_side == "right" else colors["right"],
    )

    phrase_id = int(row.get("phrase_id", 0))
    distance_pixels = int(round(float(row.get("distance_pixels", 0.0))))
    tempo_label = "BURST" if bool(row.get("tempo_burst", False)) else "calm"
    pressure = str(row.get("pressure_owner", "neutral")).upper()

    cv2.rectangle(frame, (18, 18), (400, 120), (18, 24, 31), thickness=-1)
    cv2.putText(frame, f"Phrase {phrase_id}", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 245, 248), 2)
    cv2.putText(frame, f"Distance: {distance_pixels}px", (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 228, 235), 2)
    cv2.putText(frame, f"Tempo: {tempo_label}", (30, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 228, 235), 2)
    cv2.putText(frame, f"Pressure: {pressure}", (190, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 228, 235), 2)


def _draw_fencer_box(
    frame,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    label: str,
    color: tuple[int, int, int],
) -> None:
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover
        return

    half_width = max(int(round(width / 2)), 18)
    half_height = max(int(round(height / 2)), 36)
    x1 = max(int(round(center_x)) - half_width, 0)
    y1 = max(int(round(center_y)) - half_height, 0)
    x2 = min(int(round(center_x)) + half_width, frame.shape[1] - 1)
    y2 = min(int(round(center_y)) + half_height, frame.shape[0] - 1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (int(round(center_x)), int(round(center_y))), 5, color, thickness=-1)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )


def _draw_saber_tip(
    frame,
    hand_x: float,
    hand_y: float,
    tip_x: float,
    tip_y: float,
    color: tuple[int, int, int],
) -> None:
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover
        return

    start = (int(round(hand_x)), int(round(hand_y)))
    end = (int(round(tip_x)), int(round(tip_y)))
    cv2.line(frame, start, end, color, 2)
    cv2.circle(frame, end, 6, (255, 244, 120), thickness=-1)
    cv2.circle(frame, end, 9, color, 2)
