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

The current build uses synthetic placeholder detections so the report pipeline can be developed before integrating a full CV detector.

To run the app locally once dependencies are installed:

```powershell
uvicorn api:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Hosting

The repository now includes a `Procfile` so it can be deployed to services like Railway as a simple web app.
