"""Shared pipeline pieces: watched rule, temporal split, candidate pools, MAP@5 evaluator.

Time axis: "week 11" starts Monday 2020-03-09 (first day of the log); the test weeks 31-42 start 2020-07-27.
Validation: build models from views before week 23 (2020-06-01), score against views in weeks 23-30.
Test:       build models from all views (weeks 11-30), predict weeks 31-42.
"""
import os

import numpy as np
import pandas as pd

from pathlib import Path

K = 5
ROOT = Path(__file__).resolve().parent.parent
# derived tables go to data/processed; KM_PROC points the whole pipeline to another folder (used by reproduce.ps1)
PROC = str(Path(os.environ["KM_PROC"]) if os.environ.get("KM_PROC") else ROOT / "data" / "processed") + "/"
RAW = str(ROOT / "data" / "raw") + "/"

# epoch seconds of naive local timestamps (same convention as build_interactions.py)
def _sec(s):
    return int(np.datetime64(s, "s").astype(np.int64))

W11 = _sec("2020-03-09T00:00:00")
W17 = _sec("2020-04-20T00:00:00")
W19 = _sec("2020-05-04T00:00:00")   # second (robustness) validation fold starts here
W21 = _sec("2020-05-18T00:00:00")
W23 = _sec("2020-06-01T00:00:00")   # validation window start
W25 = _sec("2020-06-15T00:00:00")
W27 = _sec("2020-06-29T00:00:00")   # ... and ends here (8 weeks)
W29 = _sec("2020-07-13T00:00:00")
W31 = _sec("2020-07-27T00:00:00")   # test window start
W43 = _sec("2020-10-19T00:00:00")   # end of test window (12 weeks)

# name: (history ends / target starts, target ends). History always starts at W11.
#   val, val2 : the two main validation folds (8-week targets)
#   val12     : 12-week target, the horizon of the Kaggle test truth (weeks 31-42), history of 8 weeks
#   val3, val5: extra folds with other cutoffs, used only to enlarge the re-ranker's training data
FOLDS = {"val": (W23, W31), "val2": (W19, W27), "val12": (W19, W31), "val3": (W21, W29), "val5": (W17, W25),
         "test": (W31, W43)}

WATCH_ABS_S = 300      # an airing counts as watched if the user spent >= 300 s on it ...
WATCH_FRAC = 0.5       # ... or >= 50% of its length, whichever is smaller


def load_interactions():
    return pd.read_parquet(PROC + "interactions.parquet")


def load_schedule():
    return pd.read_parquet(PROC + "schedule.parquet")


def test_users():
    return pd.read_csv(RAW + "submission.csv", dtype={"user_id": "uint64"})["user_id"].to_numpy()


def airings(inter, t_lo, t_hi):
    """Per (user, programme, airing): watched seconds inside the window [t_lo, t_hi), thresholded.

    The overlap interval of a view and an airing is cut at t_lo and t_hi, so a view that starts before
    the cutoff but runs past it contributes nothing after the cutoff (no future seconds in training).
    Views of one airing are summed first, because a viewer may zap away and come back.
    """
    ov_s = np.maximum(inter["view_start_s"].to_numpy(), inter["prog_start_s"].to_numpy())
    ov_e = ov_s + inter["overlap_s"].to_numpy()
    sec = np.minimum(ov_e, t_hi) - np.maximum(ov_s, t_lo)
    keep = sec > 0
    d = pd.DataFrame({
        "user_id": inter["user_id"].to_numpy()[keep],
        "tv_show_id": inter["tv_show_id"].to_numpy()[keep],
        "prog_start_s": inter["prog_start_s"].to_numpy()[keep],
        "sec": sec[keep],
        "dur": inter["prog_dur_s"].to_numpy()[keep],
    })
    g = d.groupby(["user_id", "tv_show_id", "prog_start_s"], sort=False).agg(
        sec=("sec", "sum"), dur=("dur", "first")).reset_index()
    need = np.minimum(WATCH_ABS_S, WATCH_FRAC * g["dur"].clip(lower=1))
    return g[g["sec"] >= need].drop(columns="dur")


SCREEN_MIN = 0.8       # Kaggle truth: only airings watched for >= 80% of their length count


def screen_airings(inter, t_lo, t_hi):
    """Per (user, channel, programme, airing): seconds inside [t_lo, t_hi) and the share of the airing's length."""
    ov_s = np.maximum(inter["view_start_s"].to_numpy(), inter["prog_start_s"].to_numpy())
    ov_e = ov_s + inter["overlap_s"].to_numpy()
    sec = np.minimum(ov_e, t_hi) - np.maximum(ov_s, t_lo)
    keep = sec > 0
    d = pd.DataFrame({
        "user_id": inter["user_id"].to_numpy()[keep],
        "channel_id": inter["channel_id"].to_numpy()[keep],
        "tv_show_id": inter["tv_show_id"].to_numpy()[keep],
        "prog_start_s": inter["prog_start_s"].to_numpy()[keep],
        "sec": sec[keep],
        "dur": inter["prog_dur_s"].to_numpy()[keep],
    })
    g = d.groupby(["user_id", "channel_id", "tv_show_id", "prog_start_s"], sort=False).agg(
        sec=("sec", "sum"), dur=("dur", "first")).reset_index()
    g["screen"] = g["sec"] / g["dur"].clip(lower=1)
    return g


def truth_kaggle(inter, t_lo, t_hi, k=K, min_screen=SCREEN_MIN, return_stats=False):
    """The hidden Kaggle truth rebuilt from the Data-page description (host: ogurbych).

    For each user and airing the screen time (watched time / airing length) is computed; only airings with
    screen time >= 0.8 count; per user the programmes are ranked by the number of such airings and exactly the
    top k are kept (users with fewer than k such programmes are not in the solution file). Ties at the cut are not
    specified by the host: here the programme whose first qualifying airing is earliest comes first.
    """
    g = screen_airings(inter, t_lo, t_hi)
    g = g[g["screen"] >= min_screen]
    c = g.groupby(["user_id", "tv_show_id"]).agg(n=("prog_start_s", "size"), first=("prog_start_s", "min")).reset_index()
    c = c.sort_values(["user_id", "n", "first"], ascending=[True, False, True])
    size = c.groupby("user_id")["tv_show_id"].transform("size")
    c = c[size >= k]
    rk = c.groupby("user_id").cumcount()
    top = c[rk < k]
    truth = {u: gg["tv_show_id"].tolist() for u, gg in top.groupby("user_id", sort=False)}
    if not return_stats:
        return truth
    nxt = c[rk == k].set_index("user_id")["n"]
    kth = c[rk == k - 1].set_index("user_id")["n"]
    tie = (kth.reindex(nxt.index) == nxt).sum()
    return truth, {"users": len(truth), "users_with_tie_at_cut": int(tie), "users_with_more_than_k": int(len(nxt))}


def user_item(inter, t_lo, t_hi):
    """Per (user, programme): watched seconds and number of watched airings in the window."""
    a = airings(inter, t_lo, t_hi)
    return a.groupby(["user_id", "tv_show_id"], sort=False).agg(
        sec=("sec", "sum"), n=("prog_start_s", "nunique")).reset_index()


def user_item_decay(inter, t_lo, t_hi, halflife_days):
    """Like user_item, but each airing's seconds are down-weighted by 0.5 ** (age / halflife)."""
    a = airings(inter, t_lo, t_hi)
    a["w"] = a["sec"] * 0.5 ** ((t_hi - a["prog_start_s"]) / (86400.0 * halflife_days))
    return a.groupby(["user_id", "tv_show_id"], sort=False).agg(sec=("w", "sum"), n=("prog_start_s", "nunique")).reset_index()


def candidate_pool(sched, t_lo, t_hi):
    """Labelled programmes that air in the target window (the schedule is allowed to look ahead)."""
    s = sched[(sched["start_s"] >= t_lo) & (sched["start_s"] < t_hi) & (sched["tv_show_id"] != 0)]
    return np.sort(s["tv_show_id"].unique())


def truth_top(ui, k=K):
    """The k 'most preferred' programmes per user = largest watched seconds (competition wording)."""
    ui = ui.sort_values(["user_id", "sec", "n"], ascending=[True, False, False])
    top = ui.groupby("user_id", sort=False).head(k)
    return {u: g["tv_show_id"].tolist() for u, g in top.groupby("user_id", sort=False)}


def ap_at_k(rec, truth, k=K):
    """Kaggle-style AP@k: sum of precision at hit positions / min(len(truth), k)."""
    if not truth:
        return 0.0
    truth = set(truth)
    hits, score = 0, 0.0
    for i, r in enumerate(rec[:k]):
        if r in truth:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(truth), k)


def evaluate(recs, truth, users=None):
    """MAP@5 and HitRate@5 over `users` (default: all users that have a non-empty truth)."""
    users = list(truth) if users is None else [u for u in users if u in truth]
    ap = np.array([ap_at_k(recs[u], truth[u]) for u in users])
    hit = np.array([len(set(recs[u][:K]) & set(truth[u])) > 0 for u in users])
    return {"users": len(users), "MAP@5": float(ap.mean()), "HitRate@5": float(hit.mean())}


class Split:
    """Everything a method needs for one setting ('val' or 'test')."""

    def __init__(self, mode):
        inter, sched = load_interactions(), load_schedule()
        self.mode = mode
        cut, end = FOLDS[mode]
        self.train_lo, self.train_hi = W11, cut
        self.tgt_lo, self.tgt_hi = cut, end
        self.sched = sched
        self.inter = inter
        self.train_ui = user_item(inter, self.train_lo, self.train_hi)
        self.pool = candidate_pool(sched, self.tgt_lo, self.tgt_hi)
        self.users = test_users()
        if mode != "test":
            self.truth = truth_top(user_item(inter, self.tgt_lo, self.tgt_hi))
        else:
            self.truth = None
        self.train_users = set(self.train_ui["user_id"].unique())


def pad_to_k(rec, fallback, k=K):
    """Exactly k unique ids: method output first, then the common popularity fallback."""
    out = list(dict.fromkeys(rec))[:k]
    for x in fallback:
        if len(out) >= k:
            break
        if x not in out:
            out.append(x)
    return out
