from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.analysis import analyze_video
from src.report import attach_artifact_metadata, build_report_payload
from src.visualize import render_annotated_video


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="FencingVision", version="0.1.0")
app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    athlete_side: str = Form(...),
) -> dict:
    side = athlete_side.lower()
    if side not in {"left", "right"}:
        raise HTTPException(status_code=400, detail="athlete_side must be 'left' or 'right'")

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    target_path = UPLOAD_DIR / f"latest_upload{suffix}"
    content = await video.read()
    target_path.write_bytes(content)

    analysis = analyze_video(video_path=target_path, athlete_side=side)
    annotated_video_path = UPLOAD_DIR / "latest_annotated.mp4"
    rendered_artifact = render_annotated_video(
        analysis=analysis,
        source_video_path=target_path,
        output_video_path=annotated_video_path,
    )
    payload = build_report_payload(analysis)
    artifact_url = (
        f"/uploads/{annotated_video_path.name}?t={int(time.time())}"
        if rendered_artifact
        else None
    )
    return attach_artifact_metadata(payload, artifact_url)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
