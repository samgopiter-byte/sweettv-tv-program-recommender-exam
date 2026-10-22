"""Join the view log to the broadcast schedule (same channel + interval overlap).

Output: data/processed/interactions.parquet with one row per (view, programme) overlap on labelled
airtime (tv_show_id != 0), and data/processed/schedule.parquet with the parsed schedules.
Unlabelled airtime (tv_show_id == 0) is counted in the statistics and then dropped: it is not an item.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = str(ROOT / "data" / "raw") + "/"
OUT = str(Path(os.environ["KM_PROC"]) if os.environ.get("KM_PROC") else ROOT / "data" / "processed") + "/"

SCHED_FILES = {
    "11-20": "export_arh_11-20-final.csv",
    "21-30": "export_arh_21-30-final.csv",
    "31-42": "export_arh_31-42-final.csv",
}


def load_schedule():
    parts = []
    for tag, name in SCHED_FILES.items():
        df = pd.read_csv(RAW + name, encoding="utf-8", dtype={"year_of_production": "string"})
        df["file"] = tag
        parts.append(df)
    s = pd.concat(parts, ignore_index=True)
    s["start"] = pd.to_datetime(s["start_time"], format="%d.%m.%Y %H:%M:%S")
    s["start_s"] = s["start"].astype("datetime64[s]").astype(np.int64)
    s["end_s"] = s["start_s"] + s["duration"].astype(np.int64)
    s = s.drop(columns=["start_time"]).sort_values(["channel_id", "start_s"]).reset_index(drop=True)
    return s


def load_views():
    v = pd.read_csv(RAW + "dataset11-30.csv", dtype={"user_id": "uint64", "vsetv_id": "int32"})
    v["s"] = pd.to_datetime(v["start_time"], format="%Y-%m-%d %H:%M:%S").astype("datetime64[s]").astype(np.int64)
    v["e"] = pd.to_datetime(v["stop_time"], format="%Y-%m-%d %H:%M:%S").astype("datetime64[s]").astype(np.int64)
    return v.drop(columns=["start_time", "stop_time"])


def overlaps(ps, pe, s, e):
    """Positive overlaps between programme intervals [ps, pe) (one channel, sorted by start) and views [s, e).

    Returns (index of the view, index of the programme, overlap seconds).
    """
    pe_max = np.maximum.accumulate(pe)             # protects against the 2 overlapping schedule rows
    i0 = np.searchsorted(pe_max, s, side="right")  # first programme ending after the view start
    i1 = np.searchsorted(ps, e, side="left")       # programmes starting before the view end
    cnt = np.maximum(i1 - i0, 0)
    tot = int(cnt.sum())
    if tot == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    vi = np.repeat(np.arange(len(s)), cnt)
    offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    pj = np.repeat(i0, cnt) + offs
    ov = np.minimum(e[vi], pe[pj]) - np.maximum(s[vi], ps[pj])
    ok = ov > 0
    return vi[ok], pj[ok], ov[ok]


def main():
    sched = load_schedule()
    sched.to_parquet(OUT + "schedule.parquet")
    print("schedule rows", len(sched), "labelled", int((sched["tv_show_id"] != 0).sum()))

    views = load_views()
    print("views", len(views), "period", pd.to_datetime(views["s"].min(), unit="s"), "->", pd.to_datetime(views["e"].max(), unit="s"))
    # the log covers weeks 11-30 only, so the join uses the two training schedule files
    train_sched = sched[sched["file"] != "31-42"]

    n_views_any = 0
    watch_total = 0
    watch_labelled = 0
    watch_unlabelled = 0
    chunks = []
    for ch, g in views.groupby("vsetv_id"):
        p = train_sched[train_sched["channel_id"] == ch]
        if p.empty:
            continue
        ps = p["start_s"].values
        s = g["s"].values
        vi, pj, ov = overlaps(ps, p["end_s"].values, s, g["e"].values)
        if len(vi) == 0:
            continue
        n_views_any += len(np.unique(vi))
        sid = p["tv_show_id"].values[pj]
        watch_total += int(ov.sum())
        lab = sid != 0
        watch_labelled += int(ov[lab].sum())
        watch_unlabelled += int(ov[~lab].sum())
        chunks.append(pd.DataFrame({
            "user_id": g["user_id"].values[vi[lab]],
            "channel_id": ch,
            "tv_show_id": sid[lab].astype(np.int64),
            "overlap_s": ov[lab].astype(np.int32),
            "view_start_s": s[vi[lab]],
            "prog_start_s": ps[pj[lab]],
            "prog_dur_s": p["duration"].values[pj[lab]].astype(np.int32),
        }))

    inter = pd.concat(chunks, ignore_index=True)
    inter.to_parquet(OUT + "interactions.parquet")
    print("views overlapping any schedule row:", n_views_any, f"({100*n_views_any/len(views):.2f}%)")
    print("watch seconds total / labelled / unlabelled:", watch_total, watch_labelled, watch_unlabelled,
          f"labelled share {100*watch_labelled/watch_total:.2f}%")
    print("labelled (view, programme) overlaps:", len(inter), "users", inter["user_id"].nunique(), "programmes", inter["tv_show_id"].nunique())
    print("overlap_s quantiles:", inter["overlap_s"].quantile([.1, .25, .5, .75, .9]).to_dict())


if __name__ == "__main__":
    main()
