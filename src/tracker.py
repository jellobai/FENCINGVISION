from __future__ import annotations

import pandas as pd


def track_fencers(detections: pd.DataFrame) -> pd.DataFrame:
    """Pass through placeholder tracks.

    Once a detector is integrated, this module can own identity persistence.
    """
    tracked = detections.copy()
    tracked["left_track_id"] = 1
    tracked["right_track_id"] = 2
    return tracked
