"""Quick data exploration: schedule structure, view-duration distribution, time-zone alignment of log vs schedule."""
import numpy as np
import pandas as pd

RAW = "data/raw/"
sched = pd.read_csv(RAW + "export_arh_11-20-final.csv", encoding="utf-8")
sched["start"] = pd.to_datetime(sched["start_time"], format="%d.%m.%Y %H:%M:%S")
sched["end"] = sched["start"] + pd.to_timedelta(sched["duration"], unit="s")
print("schedule", sched["start"].min(), sched["start"].max(), "channels", sched["channel_id"].nunique())
print("duration describe:\n", sched["duration"].describe())

# overlaps inside one channel
s = sched.sort_values(["channel_id", "start"])
nxt_start = s.groupby("channel_id")["start"].shift(-1)
gap = (nxt_start - s["end"]).dt.total_seconds()
print("next-start minus end: overlaps", int((gap < 0).sum()), "gaps", int((gap > 0).sum()), "exact", int((gap == 0).sum()), "of", int(gap.notna().sum()))

views = pd.read_csv(RAW + "dataset11-30.csv", nrows=2_000_000)
views["start"] = pd.to_datetime(views["start_time"])
views["stop"] = pd.to_datetime(views["stop_time"])
print("views", views["start"].min(), views["start"].max(), "users", views["user_id"].nunique(), "channels", views["vsetv_id"].nunique())
print("duration quantiles", views["duraton"].quantile([.1, .25, .5, .75, .9, .99]).to_dict())
print("hourly share of views:\n", (views["start"].dt.hour.value_counts(normalize=True).sort_index().round(3)).to_dict())

# alignment: how close do view stops sit to programme boundaries, for several clock shifts
week = views[(views["start"] >= "2020-03-09") & (views["start"] < "2020-03-16")].sample(300_000, random_state=0)
bounds = {c: np.sort(np.unique(np.r_[g["start"].values.astype("datetime64[s]").astype(np.int64), g["end"].values.astype("datetime64[s]").astype(np.int64)])) for c, g in sched.groupby("channel_id")}
week = week[week["vsetv_id"].isin(bounds)]
t = week["stop"].values.astype("datetime64[s]").astype(np.int64)
ch = week["vsetv_id"].values
for shift_h in range(-3, 4):
    d = []
    for c in np.unique(ch):
        m = ch == c
        b = bounds[c]
        x = t[m] + shift_h * 3600
        i = np.searchsorted(b, x)
        lo = b[np.clip(i - 1, 0, len(b) - 1)]
        hi = b[np.clip(i, 0, len(b) - 1)]
        d.append(np.minimum(np.abs(x - lo), np.abs(hi - x)))
    d = np.concatenate(d)
    print(f"shift {shift_h:+d}h  median dist to nearest boundary {np.median(d):7.0f}s   share within 60s {np.mean(d <= 60):.3f}")
