"""Observed distribution of the screen time (share of an airing watched) over all labelled user-airings of the log (weeks 11-30).

Justifies the "committed" threshold of 0.8 against the data and compares it with an absolute-seconds cut.
Output: results/screen_time_distribution.md
"""
import numpy as np
import pandas as pd

from common import PROC, ROOT

if __name__ == "__main__":
    inter = pd.read_parquet(PROC + "interactions.parquet")
    g = inter.groupby(["user_id", "channel_id", "tv_show_id", "prog_start_s"], sort=False).agg(
        sec=("overlap_s", "sum"), dur=("prog_dur_s", "first")).reset_index()
    g["screen"] = (g["sec"] / g["dur"].clip(lower=1)).clip(upper=1.0)
    n = len(g)
    lines = ["# Screen time of labelled user-airings (log of weeks 11-30)\n",
             f"{n:,} user-airings of programmes with an id (seconds of all overlapping views summed per airing, divided by the airing's length).\n",
             f"Median airing length {g['dur'].median():.0f} s, mean {g['dur'].mean():.0f} s; median watched seconds per user-airing {g['sec'].median():.0f} s.\n",
             "| screen time | share of user-airings |", "|---|---|"]
    edges = [0, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0001]
    labels = ["< 0.1", "0.1 - 0.3", "0.3 - 0.5", "0.5 - 0.8", "0.8 - 0.95", "0.95 - 1"]
    cut = pd.cut(g["screen"], bins=edges, right=False, labels=labels)
    for lab, c in cut.value_counts().reindex(labels).items():
        lines.append(f"| {lab} | {100 * c / n:.1f}% |")
    ge = lambda x: 100 * (g["screen"] >= x).mean()
    lines += ["", f"Share with screen time >= 0.5: {ge(0.5):.1f}%; >= 0.8 (committed): {ge(0.8):.1f}%; >= 0.9: {ge(0.9):.1f}%.", ""]
    abs300 = g["sec"] >= 300
    lines += [f"Absolute cut (>= 300 s watched): {100 * abs300.mean():.1f}% of user-airings; of these {100 * (g.loc[abs300, 'screen'] < 0.8).mean():.1f}% are below 0.8 of the airing "
              f"(median airing length of the cut's members {g.loc[abs300, 'dur'].median():.0f} s), "
              f"and {100 * ((g['screen'] >= 0.8) & ~abs300).sum() / max(1, (g['screen'] >= 0.8).sum()):.1f}% of the committed airings are shorter than 300 s of viewing.", ""]
    (ROOT / "results" / "screen_time_distribution.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
