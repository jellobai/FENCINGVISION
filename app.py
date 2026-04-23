from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.analysis import analyze_video
from src.report import build_training_report, render_phrase_table
from src.visualize import build_distance_chart, build_strip_chart


st.set_page_config(page_title="FencingVision", layout="wide")

st.title("FencingVision")
st.caption("A saber training assistant for video-driven phrase analysis.")

uploaded_file = st.file_uploader(
    "Upload a saber bout video",
    type=["mp4", "mov", "avi", "mkv"],
)
fencer_side = st.selectbox("Which side is you?", ["Left", "Right"])

st.markdown(
    """
    This beginning version focuses on:

    - phrase segmentation from motion changes
    - first-move and first-commitment estimates
    - distance and pressure trends
    - tempo-burst detection
    - training-focused summary output
    """
)

if uploaded_file is None:
    st.info("Upload a video to generate a first-pass saber training report.")
    st.stop()

work_dir = Path(".streamlit_uploads")
work_dir.mkdir(exist_ok=True)
video_path = work_dir / uploaded_file.name
video_path.write_bytes(uploaded_file.getbuffer())

analysis = analyze_video(video_path=video_path, athlete_side=fencer_side.lower())
report = build_training_report(analysis=analysis)

summary_col, metrics_col = st.columns([2, 1])

with summary_col:
    st.subheader("Training Summary")
    st.write(report.summary)
    for focus in report.practice_focus:
        st.markdown(f"- **{focus.title}**: {focus.reason}")
        st.caption(f"Practice idea: {focus.practice_idea}")

with metrics_col:
    st.subheader("Top Metrics")
    st.metric("Phrases Detected", analysis.overview.total_phrases)
    st.metric("Tempo Bursts", analysis.overview.total_tempo_bursts)
    st.metric("Near-Simultaneous Phrases", analysis.overview.near_simultaneous_phrases)
    st.metric("Delayed Commitment Flags", analysis.overview.delayed_commitment_flags)

chart_col, strip_col = st.columns(2)

with chart_col:
    st.subheader("Distance Over Time")
    st.plotly_chart(build_distance_chart(analysis.frame_features), use_container_width=True)

with strip_col:
    st.subheader("Strip Position")
    st.plotly_chart(build_strip_chart(analysis.frame_features), use_container_width=True)

st.subheader("Phrase Review")
st.dataframe(render_phrase_table(analysis.phrases), use_container_width=True)
