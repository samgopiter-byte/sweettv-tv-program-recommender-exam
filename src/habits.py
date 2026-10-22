"""Habit model: how likely is a user to be watching channel c in a given half-hour slot of the week?

The rates come from the raw view log (all airtime, labelled or not) inside the training window only.
For a programme j airing in the target window, the expected watch time of user u is

    E[u, j] = sum over airings of j, over the half-hour slots they touch,
              airing_seconds_in_slot * rate[u, channel, slot]

which is one sparse product R (users x cells) @ A.T (cells x programmes). The target-window schedule is
used only to know what airs where and when; it never reveals who watched.
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sps

from common import PROC, RAW

SLOT = 1800           # half-hour slots
SPW = 336             # slots per week
SHIFT = 144           # the epoch (Thursday 00:00) is slot 144 of a Monday-based week
POOLED = 144          # Mon-Fri share one profile, Saturday and Sunday are separate: 3 x 48 slots


def load_views():
    """Raw view log as (user_id, channel_id, s, e) epoch seconds, cached as parquet."""
    path = PROC + "views.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    v = pd.read_csv(RAW + "dataset11-30.csv", dtype={"user_id": "uint64", "vsetv_id": "int32"})
    v["s"] = pd.to_datetime(v["start_time"], format="%Y-%m-%d %H:%M:%S").astype("datetime64[s]").astype(np.int64)
    v["e"] = pd.to_datetime(v["stop_time"], format="%Y-%m-%d %H:%M:%S").astype("datetime64[s]").astype(np.int64)
    v = v[["user_id", "vsetv_id", "s", "e"]].rename(columns={"vsetv_id": "channel_id"})
    v.to_parquet(path)
    return v


def how_of(slot):
    return (slot + SHIFT) % SPW


def pooled_of(how):
    day = how // 48
    cls = np.where(day < 5, 0, day - 4)
    return cls * 48 + how % 48


def expand(s, e, t_lo, t_hi):
    """Clip intervals [s, e) to [t_lo, t_hi) and split them into half-hour slots.

    Returns (index into the input arrays, absolute slot number, overlap seconds in that slot).
    """
    s = np.maximum(s, t_lo)
    e = np.minimum(e, t_hi)
    keep = np.nonzero(e > s)[0]
    s, e = s[keep], e[keep]
    i0 = s // SLOT
    n = (e - 1) // SLOT - i0 + 1
    rep = np.repeat(np.arange(len(s)), n)
    first = np.cumsum(n) - n
    slot = i0[rep] + (np.arange(int(n.sum())) - first[rep])
    ov = np.minimum(e[rep], (slot + 1) * SLOT) - np.maximum(s[rep], slot * SLOT)
    return keep[rep], slot, ov


def decay_w(slot, t_hi, half_life):
    if half_life is None:
        return np.ones(len(slot))
    return 0.5 ** ((t_hi - (slot + 0.5) * SLOT) / (half_life * 86400.0))


def habit_scores(sp, M, half_life=None, pooled=False):
    """E[u, j] for the users of `sp` (rows, in sp.users order) and all items of M (columns)."""
    ncls = POOLED if pooled else SPW
    cls_of = (lambda h: pooled_of(h)) if pooled else (lambda h: h)
    views = load_views()
    channels = sorted(set(views["channel_id"].unique()) | set(sp.sched["channel_id"].unique()))
    chan_ix = {c: i for i, c in enumerate(channels)}
    pos = {u: i for i, u in enumerate(sp.users)}
    ncell = len(channels) * ncls

    v = views[views["user_id"].isin(sp.users) & (views["s"] < sp.train_hi) & (views["e"] > sp.train_lo)]
    rows = v["user_id"].map(pos).to_numpy()
    ch = v["channel_id"].map(chan_ix).to_numpy()
    keep, slot, ov = expand(v["s"].to_numpy(), v["e"].to_numpy(), sp.train_lo, sp.train_hi)
    w = decay_w(slot, sp.train_hi, half_life)
    col = ch[keep] * ncls + cls_of(how_of(slot))
    R = sps.csr_matrix((ov * w, (rows[keep], col)), shape=(len(pos), ncell))
    # normaliser: (decayed) seconds of the training window that fall in each slot class -> rate in [0, 1]
    all_slots = np.arange(sp.train_lo // SLOT, sp.train_hi // SLOT)
    Z = np.bincount(cls_of(how_of(all_slots)), weights=decay_w(all_slots, sp.train_hi, half_life) * SLOT, minlength=ncls)
    R.data = R.data / Z[R.indices % ncls]

    t = sp.sched[(sp.sched["tv_show_id"] != 0) & (sp.sched["start_s"] < sp.tgt_hi) & (sp.sched["end_s"] > sp.tgt_lo)]
    prog = t["tv_show_id"].map(M.item_ix)
    ok = prog.notna().to_numpy()
    t, prog = t[ok], prog[ok].astype(int).to_numpy()
    tch = t["channel_id"].map(chan_ix).to_numpy()
    keep, slot, ov = expand(t["start_s"].to_numpy(), t["end_s"].to_numpy(), sp.tgt_lo, sp.tgt_hi)
    col = tch[keep] * ncls + cls_of(how_of(slot))
    A = sps.csr_matrix((ov.astype(np.float64), (prog[keep], col)), shape=(len(M.items), ncell))
    return (R @ A.T).toarray()
