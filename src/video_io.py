from __future__ import annotations

from pathlib import Path

from src.models import VideoMetadata


def load_video_metadata(video_path: Path) -> VideoMetadata:
    try:
        import cv2  # type: ignore
    except ImportError:
        # Conservative fallback so the scaffold can still run without OpenCV.
        return VideoMetadata(
            path=video_path,
            fps=30.0,
            frame_count=900,
            width=1280,
            height=720,
            duration_seconds=30.0,
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    capture.release()

    if fps <= 1:
        fps = 30.0
    if frame_count <= 0:
        frame_count = int(fps * 30)
    if width <= 0:
        width = 1280
    if height <= 0:
        height = 720

    duration_seconds = frame_count / fps if fps else 0.0
    return VideoMetadata(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
    )
