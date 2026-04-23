from __future__ import annotations

import plotly.express as px
import pandas as pd


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
