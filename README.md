# FencingVision

FencingVision is a computer-vision project for analyzing saber bout video as a saber training assistant.

## Initial Goal

Build a beginning version that:

- takes bout video as input
- tracks both fencers over time
- estimates distance between them
- detects tempo changes from movement bursts
- flags likely attack initiations
- outputs simple charts and an annotated video

## Version 1 Scope

The first version should focus on one weapon, one stable camera angle, and rule-based event detection rather than full action classification.

## Core Stack

- Python
- OpenCV
- pose or person detection
- NumPy / pandas
- Plotly or Matplotlib
- Streamlit for demo UI

## Planned Pipeline

1. Load a bout video.
2. Detect and track each fencer across frames.
3. Compute per-frame motion features.
4. Estimate inter-fencer distance over time.
5. Flag tempo changes and likely attack starts.
6. Generate charts and a summary.

## Immediate Next Step

Create the first runnable prototype with a clean project structure and placeholder modules for detection, tracking, phrase analysis, self-review, and training recommendations.

## Current Scaffold

The current prototype includes:

- a FastAPI backend in `api.py`
- a simple responsive web frontend in `web/`
- a placeholder two-fencer detection and tracking pipeline
- frame-level feature extraction for distance, tempo, and pressure
- phrase segmentation
- saber-specific rule classification
- a training-oriented summary report

## Run Notes

The current build now supports a layered detection strategy:

- a real YOLO person detector if `ultralytics` is installed
- a foreground-contour fallback if YOLO is unavailable
- a synthetic fallback only as a last resort

To run the app locally once dependencies are installed:

```powershell
uvicorn api:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Real Detector Setup

To enable the real detector path locally, install Ultralytics in your active environment:

```powershell
pip install ultralytics
```

Then run the app normally:

```powershell
uvicorn api:app --reload
```

By default, FencingVision will try to use the YOLO model `yolo11n.pt`. You can override that with an environment variable:

```powershell
$env:FENCINGVISION_YOLO_MODEL="yolo11s.pt"
uvicorn api:app --reload
```

The results page will show which detector path was used under `Analysis Metadata`.

## Pose-Assisted Saber Tips

To improve saber-tip tracking, you can optionally install MediaPipe so the tracker can anchor the blade search from estimated wrist and elbow landmarks:

```powershell
pip install mediapipe
```

If MediaPipe is not installed, FencingVision will fall back to box-guided saber-tip estimation.

## Hosting

The repository now includes a `Procfile` so it can be deployed to services like Railway as a simple web app.
