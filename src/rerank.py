"""Candidate generation + LightGBM LambdaRank re-ranking.

Candidates per user: union of the top lists of several scorers (habit model, personal history x exposure,
conversion, ALS, association rules, popularity, CBF). Each (user, candidate) row gets scale-free features; the
label is 'candidate is one of the user's top-5 programmes by watched seconds in the target window'.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sps
import lightgbm as lgb

from common import PROC, Split, K, airings, evaluate, user_item_decay, pad_to_k
import methods as m
import habits as h

DAY = 86400
WEEK = 7 * DAY
ALS_CFG = dict(factors=32, reg=150.0, alpha=1.0, iters=20, seed=42)
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[5], learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, seed=42, lambdarank_truncation_level=10, num_threads=4,
              deterministic=True, force_row_wise=True)
VERSION = "v3"
SECONDS_SCORES = {"E_f", "E_f28", "E_f56", "E_p56", "conv"}   # scores measured in seconds over the target window


def topk_idx(S, pool_ix, k):
    """Per user: indices (on the item axis) of the k best pool items of score matrix S."""
    sub = S[:, pool_ix]
    top = np.argpartition(-sub, k - 1, axis=1)[:, :k]
    return pool_ix[top]


def rank_matrix(S, pool_ix):
    """Rank (0 = best) of every pool item within each user's row; non-pool columns get a large rank."""
    sub = S[:, pool_ix]
    order = np.argsort(-sub, axis=1, kind="stable")
    rk = np.empty(order.shape, dtype=np.int32)
    rk[np.arange(sub.shape[0])[:, None], order] = np.arange(sub.shape[1], dtype=np.int32)[None, :]
    out = np.full(S.shape, 10 ** 6, dtype=np.int32)
    out[:, pool_ix] = rk
    return out


def base_scores(sp, M):
    B = {}
    B["E_f"] = h.habit_scores(sp, M)
    B["E_f28"] = h.habit_scores(sp, M, half_life=28)
    B["E_f56"] = h.habit_scores(sp, M, half_life=56)
    B["E_p56"] = h.habit_scores(sp, M, half_life=56, pooled=True)
    B["per"] = m.personal_scores(M)
    ui28 = user_item_decay(sp.inter, sp.train_lo, sp.train_hi, 28)
    r = ui28["user_id"].map(M.user_ix).to_numpy()
    c = ui28["tv_show_id"].map(M.item_ix).to_numpy()
    P28 = sps.csr_matrix((ui28["sec"].to_numpy(), (r, c)), shape=M.sec.shape)[M.sel].toarray()
    B["per28"] = P28 / np.maximum(P28.max(axis=1, keepdims=True), 1e-9)
    B["conv"] = m.conversion_scores(M)
    B["exp"] = m.exposure_scores(M)
    B["pop"] = m.pop_scores(M)
    B["als"] = m.als_scores(M, **ALS_CFG)
    B["ar"] = m.assoc_scores(M, min_supp=10, beta=0.0)
    B["cbf"] = m.cbf_scores(M)
    return {k: v.astype(np.float32) for k, v in B.items()}


def item_table(sp, M):
    s = sp.sched[(sp.sched["tv_show_id"] != 0) & (sp.sched["start_s"] >= sp.tgt_lo) & (sp.sched["start_s"] < sp.tgt_hi)]
    weeks = (sp.tgt_hi - sp.tgt_lo) / WEEK
    g = s.groupby("tv_show_id").agg(n=("duration", "size"), air=("duration", "sum"), nch=("channel_id", "nunique"),
                                    dur=("duration", "median"), cat=("tv_show_category", "first"))
    cats = {c: i for i, c in enumerate(sorted(sp.sched["tv_show_category"].dropna().unique()))}
    T = np.zeros((len(M.items), 4), dtype=np.float32)
    catcode = np.full(len(M.items), -1, dtype=np.int64)
    for it, r in g.iterrows():
        j = M.item_ix.get(it)
        if j is None:
            continue
        T[j] = [r["n"] / weeks, r["air"] / weeks, r["nch"], r["dur"]]
        catcode[j] = cats.get(r["cat"], -1)
    return T, catcode, len(cats)


def temporal_features(sp, M, rows, cols):
    """Regularity / recency of the user's own viewing of each candidate, and recent popularity of the programme."""
    a = airings(sp.inter, sp.train_lo, sp.train_hi)
    n_weeks = (sp.train_hi - sp.train_lo) / WEEK
    a["j"] = a["tv_show_id"].map(M.item_ix)
    a = a[a["j"].notna()]
    a["j"] = a["j"].astype(np.int64)
    n_hist_users = max(a["user_id"].nunique(), 1)
    # item-level momentum over all users
    n_all = a.groupby("j")["user_id"].nunique()
    r14 = a[a["prog_start_s"] >= sp.train_hi - 14 * DAY].groupby("j")["user_id"].nunique()
    r28 = a[a["prog_start_s"] >= sp.train_hi - 28 * DAY].groupby("j")["user_id"].nunique()
    n_items = len(M.items)
    ia = np.zeros(n_items); i14 = np.zeros(n_items); i28 = np.zeros(n_items)
    ia[n_all.index.to_numpy()] = n_all.to_numpy()
    i14[r14.index.to_numpy()] = r14.to_numpy()
    i28[r28.index.to_numpy()] = r28.to_numpy()
    mom14 = i14 / (ia + 1.0)
    mom28 = i28 / (ia + 1.0)
    # user-item temporal stats for the scored users
    pos = {u: i for i, u in enumerate(sp.users)}
    a = a[a["user_id"].isin(sp.users)].copy()
    a["u"] = a["user_id"].map(pos).astype(np.int64)
    a["week"] = (a["prog_start_s"] - sp.train_lo) // WEEK
    g = a.groupby(["u", "j"]).agg(nw=("week", "nunique"), last=("prog_start_s", "max"), first=("prog_start_s", "min"),
                                  n=("prog_start_s", "nunique"))
    s14 = a[a["prog_start_s"] >= sp.train_hi - 14 * DAY].groupby(["u", "j"])["sec"].sum()
    key = rows.astype(np.int64) * n_items + cols
    gk = pd.Series(np.arange(len(g)), index=g.index.get_level_values(0).to_numpy() * n_items + g.index.get_level_values(1).to_numpy())
    idx = gk.reindex(key).to_numpy()
    seen = ~np.isnan(idx)
    ii = np.where(seen, idx, 0).astype(np.int64)
    nw = np.where(seen, g["nw"].to_numpy()[ii], 0.0)
    last = np.where(seen, g["last"].to_numpy()[ii], np.nan)
    first = np.where(seen, g["first"].to_numpy()[ii], np.nan)
    cnt = np.where(seen, g["n"].to_numpy()[ii], 0.0)
    days_since = np.where(seen, (sp.train_hi - last) / DAY, 999.0)
    span = np.where(seen, (last - first) / DAY, -1.0)
    sk = pd.Series(s14.to_numpy(), index=s14.index.get_level_values(0).to_numpy() * n_items + s14.index.get_level_values(1).to_numpy())
    s14c = sk.reindex(key).fillna(0.0).to_numpy()
    umax = pd.Series(s14c).groupby(rows).transform("max").to_numpy()
    hist_days = (sp.train_hi - sp.train_lo) / DAY
    cols_out = [nw / n_weeks, days_since, span / hist_days, cnt / n_weeks, s14c, s14c / np.maximum(umax, 1e-9),
                mom14[cols], mom28[cols], np.log(ia[cols] / n_hist_users + 1e-4)]
    names = ["reg_weeks", "days_since", "span_frac", "airings_wk", "sec14", "sec14_rel", "mom14", "mom28", "log_user_frac"]
    return np.column_stack(cols_out), names


def build(sp, M, k_each=25, k_minor=15):
    B = base_scores(sp, M)
    pool_ix = M.pool_ix
    blend = m.rownorm(B["E_f56"], pool_ix) + 1.0 * m.rownorm(B["per"] * B["exp"], pool_ix)
    sources = [(blend, k_each), (B["E_f28"], k_each), (B["per"] * B["exp"], k_each), (B["conv"], k_each),
               (B["als"], k_each), (B["pop"], k_minor), (B["ar"], k_minor), (B["cbf"], k_minor)]
    n_users = len(M.sel)
    cand = [set() for _ in range(n_users)]
    for S, k in sources:
        t = topk_idx(S, pool_ix, k)
        for i in range(n_users):
            cand[i].update(t[i].tolist())
    counts = np.array([len(c) for c in cand])
    rows = np.repeat(np.arange(n_users), counts)
    cols = np.concatenate([np.array(sorted(c)) for c in cand])
    feats, names = [], []
    tgt_weeks = (sp.tgt_hi - sp.tgt_lo) / WEEK
    # Only scale-free features: a model trained on short windows must still work when the history and the
    # target window are longer (the test setting has 20 weeks of history and a 12-week target).
    for k, v in B.items():
        raw = v[rows, cols]
        if k in SECONDS_SCORES:
            feats.append(raw / tgt_weeks); names.append(k + "_wk")       # expected seconds per week
        elif k == "cbf":
            feats.append(raw); names.append(k + "_raw")                  # cosine similarity, scale-free
        mx = np.maximum(v[:, pool_ix].max(axis=1), 1e-12)
        feats.append(raw / mx[rows]); names.append(k + "_rel")
        if k != "exp":
            feats.append(rank_matrix(v, pool_ix)[rows, cols].astype(np.float32)); names.append(k + "_rank")
    feats.append(blend[rows, cols]); names.append("blend")
    feats.append(rank_matrix(blend, pool_ix)[rows, cols].astype(np.float32)); names.append("blend_rank")
    T, catcode, ncat = item_table(sp, M)
    for q, nm in enumerate(["n_air_wk", "airtime_wk", "n_channels", "med_dur"]):
        feats.append(T[cols, q]); names.append(nm)
    feats.append(catcode[cols].astype(np.float32)); names.append("category")
    sec = M.sec[M.sel].toarray()
    tot = sec.sum(axis=1)
    weeks = (sp.train_hi - sp.train_lo) / WEEK
    feats.append((tot / weeks)[rows]); names.append("u_sec_wk")
    feats.append((sec.max(axis=1) / np.maximum(tot, 1e-9))[rows]); names.append("u_top_share")
    onehot = np.zeros((len(M.items), ncat))
    ok = catcode >= 0
    onehot[np.nonzero(ok)[0], catcode[ok]] = 1.0
    cat_share = sec @ onehot / np.maximum(tot[:, None], 1e-9)
    cc = catcode[cols]
    feats.append(np.where(cc >= 0, cat_share[rows, np.maximum(cc, 0)], 0.0)); names.append("u_cat_share")
    tf, tn = temporal_features(sp, M, rows, cols)
    feats += [tf[:, q] for q in range(tf.shape[1])]; names += tn
    X = np.column_stack(feats).astype(np.float32)
    y = None
    if sp.truth is not None:
        y = np.zeros(len(rows), dtype=np.int8)
        start = np.concatenate([[0], np.cumsum(counts)])
        for i, u in enumerate(sp.users):
            t = sp.truth.get(u)
            if t:
                y[start[i]:start[i + 1]] = np.isin(M.items[cols[start[i]:start[i + 1]]], t)
    return dict(X=X, y=y, rows=rows, cols=cols, names=np.array(names))


def get_D(mode, refresh=False):
    """Features for one setting, cached on disk. Returns (sp, M, fb, D)."""
    sp = Split(mode)
    M = m.Matrices(sp)
    fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
    path = PROC + f"rerank_{mode}_{VERSION}.npz"
    try:
        if refresh:
            raise FileNotFoundError
        z = np.load(path)                      # plain numeric/string arrays only, no pickled objects
        D = {k: z[k] for k in z.files}
        if D["y"].size == 0:
            D["y"] = None
    except FileNotFoundError:
        D = build(sp, M)
        np.savez_compressed(path, X=D["X"], y=D["y"] if D["y"] is not None else np.zeros(0, dtype=np.int8),
                            rows=D["rows"], cols=D["cols"], names=D["names"])
    return sp, M, fb, D


def fit(X, y, rows, rounds=150, params=None):
    p = dict(PARAMS)
    if params:
        p.update(params)
    order = np.argsort(rows, kind="stable")
    _, counts = np.unique(rows[order], return_counts=True)
    return lgb.train(p, lgb.Dataset(X[order], y[order], group=counts), num_boost_round=rounds)


def lists_from_scores(sp, M, fb, rows, cols, pred):
    """Top-5 per user from candidate scores; users without history -> popularity fallback; padded to 5."""
    df = pd.DataFrame({"r": rows, "c": cols, "p": pred}).sort_values(["r", "p"], ascending=[True, False])
    by = {r: g["c"].tolist() for r, g in df.groupby("r").head(K).groupby("r")}
    out = {}
    for i, u in enumerate(sp.users):
        rec = [int(M.items[j]) for j in by.get(i, [])] if M.has_hist[M.sel[i]] else []
        out[u] = pad_to_k(rec, fb)
    return out


def user_folds(n_users, folds=5, seed=0):
    """Same assignment for every setting: sp.users has the same order (template order) everywhere."""
    return np.random.default_rng(seed).integers(0, folds, size=n_users)


def crossfit_pooled(Ds, n_users, folds=5, rounds=150, params=None, seed=0):
    """Train on the other users' rows of all settings in Ds; predict the held-out users' rows of each setting."""
    fo = user_folds(n_users, folds, seed)
    preds = [np.zeros(len(D["rows"])) for D in Ds]
    for f in range(folds):
        X = np.vstack([D["X"][fo[D["rows"]] != f] for D in Ds])
        y = np.concatenate([D["y"][fo[D["rows"]] != f] for D in Ds])
        # row ids must stay unique per (setting, user): offset them
        r = np.concatenate([D["rows"][fo[D["rows"]] != f] + k * n_users for k, D in enumerate(Ds)])
        model = fit(X, y, r, rounds, params)
        for k, D in enumerate(Ds):
            te = fo[D["rows"]] == f
            preds[k][te] = model.predict(D["X"][te])
    return preds, model
