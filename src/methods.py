"""Recommendation methods as score matrices (test users x items), plus the common top-5 selection.

Shared protocol: candidates = Split.pool (programmes airing in the target window); repeats of already
watched programmes are allowed unless `repeats=False`; users with no usable training history get the
common popularity fallback, and every list is padded to exactly 5 unique ids with the same fallback.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import K, pad_to_k


def pop_ranking(train_ui, pool, by="seconds"):
    ui = train_ui[train_ui["tv_show_id"].isin(pool)]
    sc = ui.groupby("tv_show_id")["sec"].sum() if by == "seconds" else ui.groupby("tv_show_id")["user_id"].nunique()
    return sc.sort_values(ascending=False).index.tolist()


class Matrices:
    """Index maps and user x item matrices shared by all methods."""

    def __init__(self, sp):
        self.items = np.array(sorted(set(sp.train_ui["tv_show_id"]) | set(sp.pool)))
        self.item_ix = {it: i for i, it in enumerate(self.items)}
        self.users = np.array(sorted(set(sp.train_ui["user_id"]) | set(sp.users)), dtype=np.uint64)
        self.user_ix = {u: i for i, u in enumerate(self.users)}
        ui = sp.train_ui
        r = ui["user_id"].map(self.user_ix).to_numpy()
        c = ui["tv_show_id"].map(self.item_ix).to_numpy()
        shape = (len(self.users), len(self.items))
        self.sec = sps.csr_matrix((ui["sec"].to_numpy().astype(np.float32), (r, c)), shape=shape)
        self.minutes = sps.csr_matrix((np.log1p(ui["sec"].to_numpy() / 60.0), (r, c)), shape=shape)
        self.binary = sps.csr_matrix((np.ones(len(ui), dtype=np.float32), (r, c)), shape=shape)
        self.pool_ix = np.array([self.item_ix[i] for i in sp.pool])
        self.has_hist = np.asarray(self.binary.sum(axis=1)).ravel() > 0
        self.sel = np.array([self.user_ix[u] for u in sp.users])      # rows of the users we score
        self.sp = sp


# ---- score matrices: rows = sp.users (in order), columns = M.items ---------------------------------
def pop_scores(M):
    ui = M.sp.train_ui
    s = ui.groupby("tv_show_id")["sec"].sum()
    v = np.zeros(len(M.items), dtype=np.float32)
    for it, x in s.items():
        v[M.item_ix[it]] = x
    v /= max(v[M.pool_ix].max(), 1e-9)
    return np.tile(v, (len(M.sel), 1))


def personal_scores(M):
    """The user's own watched seconds, scaled by the user's maximum (repeat consumption)."""
    S = M.sec[M.sel].toarray()
    return S / np.maximum(S.max(axis=1, keepdims=True), 1e-9)


def als_scores(M, factors=32, reg=10.0, alpha=1.0, iters=20, seed=42, weight="log"):
    from implicit.als import AlternatingLeastSquares
    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters, random_state=seed,
                                    calculate_training_loss=False)
    conf = (M.minutes if weight == "log" else M.binary).copy()
    conf.data = 1.0 + alpha * conf.data
    model.fit(conf, show_progress=False)
    return np.asarray(model.user_factors[M.sel] @ model.item_factors.T)


def assoc_scores(M, min_supp=20, beta=0.0):
    """Pairwise rules i -> j over each user's training history (the basket).

    confidence(i->j) = supp(i,j) / supp(i); the rule weight is confidence / (supp(j)/N)**beta, so
    beta = 0 is plain confidence and beta = 1 is lift. A user's score for j sums the weights of the
    rules whose antecedent i is in the user's history. Rules with supp(i,j) < min_supp are dropped.
    """
    B = M.binary
    N = B.shape[0]
    C = (B.T @ B).toarray().astype(np.float32)
    n = np.diag(C).copy()
    np.fill_diagonal(C, 0)
    C[C < min_supp] = 0
    R = C / np.maximum(n[:, None], 1) / np.maximum(n[None, :] / N, 1e-9) ** beta
    return np.asarray(B[M.sel] @ R)


def cbf_scores(M, weights="log"):
    X = item_features(M.sp.sched, M.items)
    W = M.minutes if weights == "log" else M.binary
    P = normalize(W[M.sel] @ X)
    return (P @ X.T).toarray()


def item_features(sched, items):
    s = sched[sched["tv_show_id"] != 0].sort_values("start_s")
    meta = s.groupby("tv_show_id").agg({
        "tv_show_category": "first", "tv_show_genre_1": "first", "tv_show_genre_2": "first",
        "tv_show_genre_3": "first", "year_of_production": "first", "director": "first", "actors": "first",
    })
    docs = []
    for it in items:
        toks = []
        if it in meta.index:
            r = meta.loc[it]
            if isinstance(r["tv_show_category"], str):
                toks.append("cat_" + r["tv_show_category"])
            for g in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
                if isinstance(r[g], str):
                    toks.append("g_" + r[g])
            y = r["year_of_production"]
            if isinstance(y, str) and y[:4].isdigit():
                toks.append("dec_" + y[:3])
            for fld, pre in (("director", "dir_"), ("actors", "act_")):
                if isinstance(r[fld], str):
                    toks += [pre + t.strip().replace(" ", "_") for t in r[fld].split(",") if t.strip()]
        docs.append(toks)
    vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True)
    return normalize(vec.fit_transform(docs))


# ---- combine + select top-5 ----------------------------------------------------------------------
def rownorm(S, pool_ix):
    """Scale each user's scores to [0, 1] over the candidate pool (min-max), so methods can be blended."""
    sub = S[:, pool_ix]
    lo, hi = sub.min(axis=1, keepdims=True), sub.max(axis=1, keepdims=True)
    out = np.zeros_like(S, dtype=np.float32)
    out[:, pool_ix] = (sub - lo) / np.maximum(hi - lo, 1e-9)
    return out


def blend(M, parts):
    """parts: list of (weight, score matrix). Weighted sum of row-normalised scores."""
    S = np.zeros((len(M.sel), len(M.items)), dtype=np.float32)
    for w, P in parts:
        S += w * rownorm(P, M.pool_ix)
    return S


def recommend(M, S, fallback, repeats=True):
    sub = S[:, M.pool_ix].copy()
    if not repeats:
        sub[M.binary[M.sel].toarray()[:, M.pool_ix] > 0] = -np.inf
    top = np.argpartition(-sub, K - 1, axis=1)[:, :K]
    order = np.take_along_axis(sub, top, axis=1).argsort(axis=1)[:, ::-1]
    ix = M.pool_ix[np.take_along_axis(top, order, axis=1)]
    out = {}
    for row, u in enumerate(M.sp.users):
        rec = [int(M.items[j]) for j in ix[row]] if M.has_hist[M.sel[row]] else []
        out[u] = pad_to_k(rec, fallback)
    return out


def with_input(M, X):
    """Same index maps as M, but with the user x item matrices built from X (committed-airing counts), so that ALS,
    association rules and CBF can be fed the truth-aligned signal instead of watched seconds."""
    import copy
    M2 = copy.copy(M)
    X = X.tocsr().astype(np.float32)
    L = X.copy()
    L.data = np.log1p(L.data)
    M2.minutes = L
    M2.binary = (X > 0).astype(np.float32)
    M2.sec = X
    M2.has_hist = np.asarray(M2.binary.sum(axis=1)).ravel() > 0
    return M2


def ppr_scores(M, X, d=0.85, iters=15):
    """Personalised PageRank on a programme co-view graph.

    Graph: item-item cosine similarity of the log-weighted user x item matrix X (all users of the log); every row is
    normalised to transition probabilities. The restart distribution of a user is the user's own normalised weights.
    Iteration: r <- (1 - d) * p0 + d * r @ P  (d = damping, 0.85 is the classical value).
    """
    Xw = X.tocsr().astype(np.float32)
    Xw.data = np.log1p(Xw.data)
    Xn = normalize(Xw, axis=0)                           # L2-normalise the item columns
    S = (Xn.T @ Xn).toarray().astype(np.float32)         # cosine similarity between programmes
    np.fill_diagonal(S, 0.0)
    S /= np.maximum(S.sum(axis=1, keepdims=True), 1e-9)  # P[i, j]: probability to move from programme i to j
    p0 = Xw[M.sel].toarray()
    p0 /= np.maximum(p0.sum(axis=1, keepdims=True), 1e-9)
    r = p0.copy()
    for _ in range(iters):
        r = (1.0 - d) * p0 + d * (r @ S)
    return r


def conversion_scores(M, c=3600.0):
    """Programme-specific loyalty: the user's training seconds on j divided by the airtime of j in training,
    times the airtime of j in the target window (c smooths programmes with little airtime)."""
    sp = M.sp
    s = sp.sched[sp.sched["tv_show_id"] != 0]
    tr = s[(s["start_s"] >= sp.train_lo) & (s["start_s"] < sp.train_hi)].groupby("tv_show_id")["duration"].sum()
    tg = s[(s["start_s"] >= sp.tgt_lo) & (s["start_s"] < sp.tgt_hi)].groupby("tv_show_id")["duration"].sum()
    t_tr = np.zeros(len(M.items))
    t_tg = np.zeros(len(M.items))
    for it, x in tr.items():
        if it in M.item_ix:
            t_tr[M.item_ix[it]] = x
    for it, x in tg.items():
        if it in M.item_ix:
            t_tg[M.item_ix[it]] = x
    S = M.sec[M.sel].toarray()
    return S / (t_tr[None, :] + c) * t_tg[None, :]


def exposure_scores(M):
    """Airtime (seconds) of each programme in the target window, from the schedule, log-scaled to [0, 1].

    The schedule of the target window may be used to know what airs when; it never reveals who watched.
    """
    sp = M.sp
    s = sp.sched
    s = s[(s["start_s"] >= sp.tgt_lo) & (s["start_s"] < sp.tgt_hi) & (s["tv_show_id"] != 0)]
    a = s.groupby("tv_show_id")["duration"].sum()
    v = np.zeros(len(M.items), dtype=np.float32)
    for it, x in a.items():
        if it in M.item_ix:
            v[M.item_ix[it]] = np.log1p(x)
    v /= max(v[M.pool_ix].max(), 1e-9)
    return np.tile(v, (len(M.sel), 1))
