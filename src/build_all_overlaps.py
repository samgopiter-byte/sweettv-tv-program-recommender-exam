"""Overlaps of the template users' views with ALL airings (labelled and unlabelled) of the training schedules.

Needed by the commitment-habit model, which estimates how likely a user is to watch >= 80% of an airing that
starts in a given channel / half-hour slot of the week, whether or not the airing has a programme id.
Output: data/processed/all_overlaps.parquet (same columns as interactions.parquet; tv_show_id == 0 for unlabelled).
"""
import numpy as np
import pandas as pd

from build_interactions import OUT, load_schedule, load_views, overlaps
from common import test_users

if __name__ == "__main__":
    sched = load_schedule()
    train_sched = sched[sched["file"] != "31-42"]
    views = load_views()
    views = views[views["user_id"].isin(test_users())]
    print("template users' views:", len(views))
    parts = []
    for ch, g in views.groupby("vsetv_id"):
        p = train_sched[train_sched["channel_id"] == ch]
        if p.empty:
            continue
        ps = p["start_s"].values
        s = g["s"].values
        vi, pj, ov = overlaps(ps, p["end_s"].values, s, g["e"].values)
        if len(vi) == 0:
            continue
        parts.append(pd.DataFrame({
            "user_id": g["user_id"].values[vi],
            "channel_id": np.int32(ch),
            "tv_show_id": p["tv_show_id"].values[pj].astype(np.int64),
            "overlap_s": ov.astype(np.int32),
            "view_start_s": s[vi],
            "prog_start_s": ps[pj],
            "prog_dur_s": p["duration"].values[pj].astype(np.int32),
        }))
    allov = pd.concat(parts, ignore_index=True)
    allov.to_parquet(OUT + "all_overlaps.parquet")
    lab = allov["tv_show_id"] != 0
    print("rows", len(allov), "| labelled", int(lab.sum()), "| unlabelled", int((~lab).sum()), "| users", allov["user_id"].nunique())
