"""High-screen-time ("committed viewing") tools built from the Kaggle Data-page description of the hidden truth.

Truth (host): for each user and airing, screen time = watched time / airing length; only airings with screen time
>= 0.8 count; per user the programmes are ranked by the number of such airings; exactly the top 5 are kept.

* tie_info / truth_samples / evaluate_samples : tie-aware local Kaggle replica. Many users have ties at the 5th
  place and the host does not say how they were broken, so the score is the expectation over random tie-breaks
  (the same random draws for every method, so methods are compared on identical truths).
* hs_scores : scorers aligned with that truth (counts of committed airings, conversion, slot-based habit).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sps

from common import K, SCREEN_MIN, ap_at_k, screen_airings
import habits as h

DAY = 86400
WEEK = 7 * DAY


# ---- truth with ties ------------------------------------------------------------------------------------
def qualifying_counts(inter, t_lo, t_hi):
    """Per (user, programme): number of airings watched for >= 80% inside the window, and the first such airing."""
    g = screen_airings(inter, t_lo, t_hi)
    g = g[g["screen"] >= SCREEN_MIN]
    return g.groupby(["user_id", "tv_show_id"]).agg(n=("prog_start_s", "size"), first=("prog_start_s", "min")).reset_index()


def tie_info(counts, k=K):
    """user -> (programmes strictly above the cut, programmes tied at the cut, places left for the tied, deterministic top-k)."""
    out = {}
    c = counts.sort_values(["user_id", "n", "first"], ascending=[True, False, True])
    for u, g in c.groupby("user_id", sort=False):
        if len(g) < k:
            continue
        n, items = g["n"].to_numpy(), g["tv_show_id"].to_numpy()
        c5 = n[k - 1]
        sure = items[n > c5].tolist()
        tied = items[n == c5].tolist()
        out[u] = (sure, tied, k - len(sure), items[:k].tolist())
    return out


def truth_samples(info, draws=40, seed=0):
    """user -> list of possible truth sets (random tie-breaks); identical for every method."""
    rng = np.random.default_rng(seed)
    out = {}
    for u, (sure, tied, need, det) in info.items():
        if need >= len(tied):
            out[u] = [frozenset(sure) | frozenset(tied)]
        else:
            out[u] = [frozenset(sure) | frozenset(rng.choice(tied, need, replace=False).tolist()) for _ in range(draws)]
    return out


def evaluate_samples(recs, samples, users, info=None):
    """Expected MAP@5 over the random tie-breaks (and, if info is given, MAP@5 for the deterministic earliest-first truth)."""
    us = [u for u in users if u in samples]
    exp_ap = np.array([np.mean([ap_at_k(recs[u], t) for t in samples[u]]) for u in us])
    res = {"users": len(us), "MAP@5": float(exp_ap.mean())}
    if info is not None:
        det = np.array([ap_at_k(recs[u], info[u][3]) for u in us])
        res["MAP@5_det"] = float(det.mean())
    return res


# ---- scorers aligned with the truth ------------------------------------------------------------------------
def hs_matrices(sp, M, halflives=(None, 14, 28, 56)):
    """Dense (users x items) counts of committed airings in the history, optionally recency-weighted."""
    g = screen_airings(sp.inter, sp.train_lo, sp.train_hi)
    g = g[(g["screen"] >= SCREEN_MIN) & g["user_id"].isin(sp.users)]
    r = g["user_id"].map(M.user_ix).to_numpy()
    c = g["tv_show_id"].map(M.item_ix).to_numpy()
    ok = ~(pd.isna(r) | pd.isna(c))
    r, c, age = r[ok].astype(int), c[ok].astype(int), (sp.train_hi - g["prog_start_s"].to_numpy()[ok]) / DAY
    out = {}
    for hl in halflives:
        w = np.ones(len(r)) if hl is None else 0.5 ** (age / hl)
        S = sps.csr_matrix((w, (r, c)), shape=M.sec.shape)[M.sel].toarray().astype(np.float32)
        out["hs_cnt" if hl is None else f"hs_cnt{hl}"] = S
    return out


def committed_matrix_all(sp, M):
    """Sparse (all users of the log x items): number of airings watched for >= 80% in the history window."""
    g = screen_airings(sp.inter, sp.train_lo, sp.train_hi)
    g = g[g["screen"] >= SCREEN_MIN]
    r, c = g["user_id"].map(M.user_ix), g["tv_show_id"].map(M.item_ix)
    ok = (r.notna() & c.notna()).to_numpy()
    return sps.csr_matrix((np.ones(int(ok.sum()), dtype=np.float32), (r[ok].astype(int).to_numpy(), c[ok].astype(int).to_numpy())),
                          shape=M.sec.shape)


def airing_counts(sp, M, t_lo, t_hi):
    """Number of airings (schedule rows) of every programme starting inside the window."""
    s = sp.sched[(sp.sched["tv_show_id"] != 0) & (sp.sched["start_s"] >= t_lo) & (sp.sched["start_s"] < t_hi)]
    cnt = s.groupby("tv_show_id").size()
    v = np.zeros(len(M.items), dtype=np.float32)
    for it, x in cnt.items():
        j = M.item_ix.get(it)
        if j is not None:
            v[j] = x
    return v


def conversion_hs(sp, M, hs_all, c=2.0):
    """Expected committed airings: (user's committed airings of j / airings of j in history) x airings of j in the target."""
    n_hist = airing_counts(sp, M, sp.train_lo, sp.train_hi)
    n_tgt = airing_counts(sp, M, sp.tgt_lo, sp.tgt_hi)
    return hs_all / (n_hist[None, :] + c) * n_tgt[None, :]


def habit_hs(sp, M, allov, half_life=None, pooled=False, shrink=1.0, kind=None):
    """Slot-based commitment habit: P(watch >= 80% of an airing starting in this channel / time slot), summed over
    the airings of each programme in the target window (= expected number of committed airings).

    kind: 'fine' half-hour of the week (336 slots), 'pooled' Mon-Fri / Sat / Sun x half-hour (144), 'hour' hour of the
    week (168), 'channel' no time at all (the user's rate on the channel). `pooled=True` means kind='pooled'.
    """
    kind = kind or ("pooled" if pooled else "fine")
    ncls = {"fine": h.SPW, "pooled": h.POOLED, "hour": 168, "channel": 1}[kind]
    channels = sorted(set(sp.sched["channel_id"].unique()) | set(allov["channel_id"].unique()))
    chan_ix = {c: i for i, c in enumerate(channels)}
    ncell = len(channels) * ncls
    pos = {u: i for i, u in enumerate(sp.users)}

    def cls_of(start_s):
        start_s = np.asarray(start_s)
        if kind == "channel":
            return np.zeros(len(start_s), dtype=np.int64)
        if kind == "hour":
            return ((start_s + 1800) // 3600 + 72) % 168                   # nearest hour; the epoch is Thursday 00:00
        how = h.how_of((start_s + h.SLOT // 2) // h.SLOT)                  # nearest half hour
        return h.pooled_of(how) if kind == "pooled" else how

    def cell_of(channel, start_s):
        return np.asarray(channel) * ncls + cls_of(start_s)

    def decay(start_s):
        if half_life is None:
            return np.ones(len(start_s))
        return 0.5 ** ((sp.train_hi - np.asarray(start_s)) / (half_life * DAY))

    a = screen_airings(allov, sp.train_lo, sp.train_hi)
    a = a[(a["screen"] >= SCREEN_MIN) & a["user_id"].isin(sp.users)]
    ch = a["channel_id"].map(chan_ix).to_numpy().astype(int)
    R = sps.csr_matrix((decay(a["prog_start_s"].to_numpy()), (a["user_id"].map(pos).to_numpy().astype(int), cell_of(ch, a["prog_start_s"].to_numpy()))),
                       shape=(len(pos), ncell))
    hist = sp.sched[(sp.sched["start_s"] >= sp.train_lo) & (sp.sched["start_s"] < sp.train_hi)]
    hc = hist["channel_id"].map(chan_ix).to_numpy().astype(int)
    Z = np.bincount(cell_of(hc, hist["start_s"].to_numpy()), weights=decay(hist["start_s"].to_numpy()), minlength=ncell)
    R.data = R.data / (Z[R.indices] + shrink)
    t = sp.sched[(sp.sched["tv_show_id"] != 0) & (sp.sched["start_s"] >= sp.tgt_lo) & (sp.sched["start_s"] < sp.tgt_hi)]
    prog = t["tv_show_id"].map(M.item_ix)
    ok = prog.notna().to_numpy()
    t, prog = t[ok], prog[ok].astype(int).to_numpy()
    tc = t["channel_id"].map(chan_ix).to_numpy().astype(int)
    A = sps.csr_matrix((np.ones(len(t)), (prog, cell_of(tc, t["start_s"].to_numpy()))), shape=(len(M.items), ncell))
    return (R @ A.T).toarray().astype(np.float32)


def hs_scores(sp, M, allov):
    """All truth-aligned scorers as dense (users x items) matrices."""
    B = hs_matrices(sp, M)
    B["hs_conv"] = conversion_hs(sp, M, B["hs_cnt"])
    B["hs_conv56"] = conversion_hs(sp, M, B["hs_cnt56"])
    B["hab_hs"] = habit_hs(sp, M, allov)
    B["hab_hs56"] = habit_hs(sp, M, allov, half_life=56)
    B["hab_hs_p"] = habit_hs(sp, M, allov, pooled=True)
    B["hab_hs_p56"] = habit_hs(sp, M, allov, half_life=56, pooled=True)
    return B


def hs_scores_extra(sp, M, allov):
    """Variants added in the second iteration: other recency half-lives, hour-of-week and channel-only habits."""
    B = {}
    B["hab_hs_p28"] = habit_hs(sp, M, allov, half_life=28, pooled=True)
    B["hab_hs_p112"] = habit_hs(sp, M, allov, half_life=112, pooled=True)
    B["hab_hs_h56"] = habit_hs(sp, M, allov, half_life=56, kind="hour")
    B["hab_hs_ch56"] = habit_hs(sp, M, allov, half_life=56, kind="channel")
    return B


def commit_priors(sp, M):
    """Programme-level propensity to be watched to the end (all users of the log), scale-free:
    commit_rate = committed (user, airing) pairs per airing and active user; users28 = share of active users with a
    committed airing of the programme in the last 28 days of the history."""
    g = screen_airings(sp.inter, sp.train_lo, sp.train_hi)
    active = max(g["user_id"].nunique(), 1)
    c = g[g["screen"] >= SCREEN_MIN]
    c = c.assign(j=c["tv_show_id"].map(M.item_ix))
    c = c[c["j"].notna()]
    c["j"] = c["j"].astype(int)
    n_hist = airing_counts(sp, M, sp.train_lo, sp.train_hi)
    pairs = np.bincount(c["j"].to_numpy(), minlength=len(M.items)).astype(np.float64)
    recent = c[c["prog_start_s"] >= sp.train_hi - 28 * DAY]
    u28 = recent.groupby("j")["user_id"].nunique()
    users28 = np.zeros(len(M.items))
    users28[u28.index.to_numpy()] = u28.to_numpy()
    return pairs / (n_hist + 1.0) / active, users28 / active
