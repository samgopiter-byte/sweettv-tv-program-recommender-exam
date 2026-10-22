"""Truth-aligned re-ranker data (v4): the v3 features plus committed-airing scorers/features; labels come from the
Kaggle replica (airings watched >= 80%, programmes ranked by their number).

Per (user, candidate) row three labels are kept:
  y_cnt  committed airings of the candidate in the target window, per week      (Poisson regression)
  y_bin  1 if the candidate is in the deterministic earliest-first top-5          (LambdaRank)
  y_soft probability of being in the top-5 under random tie-breaks                (graded LambdaRank / cross-entropy)
and `elig` marks the users that have a truth at all (>= 5 committed programmes in the target window).
"""
import numpy as np
import pandas as pd

import hs
import methods as m
import rerank as rr
from common import PROC, Split, K, SCREEN_MIN, screen_airings

VERSION2 = "v6"
DAY, WEEK = 86400, 7 * 86400
HS_PAST = ("hs_cnt", "hs_cnt14", "hs_cnt28", "hs_cnt56")                      # committed airings in the history
HS_EXPECTED = ("hs_conv", "hs_conv56", "hab_hs", "hab_hs56", "hab_hs_p", "hab_hs_p56",
               "hab_hs_p28", "hab_hs_p112", "hab_hs_h56", "hab_hs_ch56")       # expected in the target


def hs_temporal(sp, M, rows, cols):
    """Regularity / recency of committed viewing of each candidate (airings watched >= 80%)."""
    g = screen_airings(sp.inter, sp.train_lo, sp.train_hi)
    g = g[(g["screen"] >= SCREEN_MIN) & g["user_id"].isin(sp.users)].copy()
    pos = {u: i for i, u in enumerate(sp.users)}
    g["u"] = g["user_id"].map(pos).astype(np.int64)
    g["j"] = g["tv_show_id"].map(M.item_ix)
    g = g[g["j"].notna()]
    g["j"] = g["j"].astype(np.int64)
    n_items = len(M.items)
    g["week"] = (g["prog_start_s"] - sp.train_lo) // WEEK
    agg = g.groupby(["u", "j"]).agg(nw=("week", "nunique"), last=("prog_start_s", "max"), first=("prog_start_s", "min"))
    last14 = g[g["prog_start_s"] >= sp.train_hi - 14 * DAY].groupby(["u", "j"]).size()
    key = rows.astype(np.int64) * n_items + cols
    ak = pd.Series(np.arange(len(agg)), index=agg.index.get_level_values(0).to_numpy() * n_items + agg.index.get_level_values(1).to_numpy())
    idx = ak.reindex(key).to_numpy()
    seen = ~np.isnan(idx)
    ii = np.where(seen, idx, 0).astype(np.int64)
    n_weeks = (sp.train_hi - sp.train_lo) / WEEK
    nw = np.where(seen, agg["nw"].to_numpy()[ii], 0.0) / n_weeks
    days = np.where(seen, (sp.train_hi - agg["last"].to_numpy()[ii]) / DAY, 999.0)
    span = np.where(seen, (agg["last"].to_numpy()[ii] - agg["first"].to_numpy()[ii]) / DAY, -1.0) / ((sp.train_hi - sp.train_lo) / DAY)
    lk = pd.Series(last14.to_numpy(), index=last14.index.get_level_values(0).to_numpy() * n_items + last14.index.get_level_values(1).to_numpy())
    l14 = lk.reindex(key).fillna(0.0).to_numpy()
    return np.column_stack([nw, days, span, l14]), ["hs_reg_weeks", "hs_days_since", "hs_span_frac", "hs_last14"]


def build2(sp, M, allov, k_each=25, k_minor=15):
    B = rr.base_scores(sp, M)
    B.update(hs.hs_scores(sp, M, allov))
    B.update(hs.hs_scores_extra(sp, M, allov))
    # the four methods of the course fed with the truth-aligned signal (committed airings of all users of the log)
    X_all = hs.committed_matrix_all(sp, M)
    Mc = m.with_input(M, X_all)
    B["als_c"] = m.als_scores(Mc, factors=32, reg=50.0, alpha=5.0)
    B["ar_c"] = m.assoc_scores(Mc, min_supp=10, beta=0.3)
    B["cbf_c"] = m.cbf_scores(Mc)
    B["ppr"] = m.ppr_scores(M, X_all, d=0.85)
    B["ppr50"] = m.ppr_scores(M, X_all, d=0.5)
    B = {k: v.astype(np.float32) for k, v in B.items()}
    pool_ix = M.pool_ix
    n_users = len(M.sel)
    pop_n = m.rownorm(B["pop"], pool_ix)
    blend = m.rownorm(B["E_f56"], pool_ix) + m.rownorm(B["per"] * B["exp"], pool_ix)
    sources = [(B["hab_hs_p"], k_each), (B["hab_hs56"], k_each), (B["hs_conv"], k_each), (B["hs_cnt56"], k_each),
               (B["hs_cnt28"], k_each), (blend, k_each), (B["als"], k_each), (B["pop"], k_minor), (B["ar"], k_minor),
               (B["hab_hs_h56"], k_minor), (B["hab_hs_ch56"], k_minor), (B["ppr"], k_each), (B["als_c"], k_minor)]
    cand = [set() for _ in range(n_users)]
    for S, k in sources:
        mx = np.maximum(S[:, pool_ix].max(axis=1, keepdims=True), 1e-12)
        t = rr.topk_idx(S / mx + 1e-6 * pop_n, pool_ix, k)        # popularity breaks ties among zero scores
        for i in range(n_users):
            cand[i].update(t[i].tolist())
    counts = np.array([len(c) for c in cand])
    rows = np.repeat(np.arange(n_users), counts)
    cols = np.concatenate([np.array(sorted(c)) for c in cand])
    tgt_weeks = (sp.tgt_hi - sp.tgt_lo) / WEEK
    hist_weeks = (sp.train_hi - sp.train_lo) / WEEK
    feats, names = [], []
    for k, v in B.items():
        raw = v[rows, cols]
        if k in rr.SECONDS_SCORES or k in HS_EXPECTED:
            feats.append(raw / tgt_weeks); names.append(k + "_wk")
        elif k in HS_PAST:
            feats.append(raw / hist_weeks); names.append(k + "_wk")
        elif k == "cbf":
            feats.append(raw); names.append(k + "_raw")
        mx = np.maximum(v[:, pool_ix].max(axis=1), 1e-12)
        feats.append(raw / mx[rows]); names.append(k + "_rel")
        if k != "exp":
            feats.append(rr.rank_matrix(v, pool_ix)[rows, cols].astype(np.float32)); names.append(k + "_rank")
    feats.append(blend[rows, cols]); names.append("blend")
    T, catcode, ncat = rr.item_table(sp, M)
    for q, nm in enumerate(["n_air_wk", "airtime_wk", "n_channels", "med_dur"]):
        feats.append(T[cols, q]); names.append(nm)
    n_hist = hs.airing_counts(sp, M, sp.train_lo, sp.train_hi)
    feats.append(n_hist[cols] / hist_weeks); names.append("n_hist_air_wk")
    pc_rate, pc_users28 = hs.commit_priors(sp, M)
    feats.append(pc_rate[cols]); names.append("pop_commit_rate")
    feats.append(pc_users28[cols]); names.append("pop_commit_users28")
    hs_tot = B["hs_cnt"].sum(axis=1)
    feats.append((hs_tot / hist_weeks)[rows]); names.append("u_hs_wk")
    feats.append((B["hs_cnt"].max(axis=1) / np.maximum(hs_tot, 1e-9))[rows]); names.append("u_hs_top_share")
    feats.append(catcode[cols].astype(np.float32)); names.append("category")
    sec = M.sec[M.sel].toarray()
    tot = sec.sum(axis=1)
    feats.append((tot / hist_weeks)[rows]); names.append("u_sec_wk")
    feats.append((sec.max(axis=1) / np.maximum(tot, 1e-9))[rows]); names.append("u_top_share")
    onehot = np.zeros((len(M.items), ncat))
    ok = catcode >= 0
    onehot[np.nonzero(ok)[0], catcode[ok]] = 1.0
    cat_share = sec @ onehot / np.maximum(tot[:, None], 1e-9)
    cc = catcode[cols]
    feats.append(np.where(cc >= 0, cat_share[rows, np.maximum(cc, 0)], 0.0)); names.append("u_cat_share")
    tf, tn = rr.temporal_features(sp, M, rows, cols)
    feats += [tf[:, q] for q in range(tf.shape[1])]; names += tn
    tf2, tn2 = hs_temporal(sp, M, rows, cols)
    feats += [tf2[:, q] for q in range(tf2.shape[1])]; names += tn2
    X = np.column_stack(feats).astype(np.float32)
    out = dict(X=X, rows=rows, cols=cols, names=np.array(names))
    if sp.truth is not None:
        cnt = hs.qualifying_counts(sp.inter, sp.tgt_lo, sp.tgt_hi)
        info = hs.tie_info(cnt)
        n_items = len(M.items)
        pos = {u: i for i, u in enumerate(sp.users)}
        c_u = cnt["user_id"].map(pos)
        c_j = cnt["tv_show_id"].map(M.item_ix)
        okc = c_u.notna() & c_j.notna()
        ck = pd.Series(cnt["n"].to_numpy()[okc.to_numpy()], index=c_u[okc].astype(np.int64).to_numpy() * n_items + c_j[okc].astype(np.int64).to_numpy())
        key = rows.astype(np.int64) * n_items + cols
        y_cnt = ck.reindex(key).fillna(0.0).to_numpy() / tgt_weeks
        y_bin = np.zeros(len(rows), dtype=np.int8)
        y_soft = np.zeros(len(rows), dtype=np.float32)
        elig = np.zeros(len(rows), dtype=bool)
        start = np.concatenate([[0], np.cumsum(counts)])
        for i, u in enumerate(sp.users):
            if u not in info:
                continue
            sure, tied, need, det = info[u]
            sl = slice(start[i], start[i + 1])
            items = M.items[cols[sl]]
            elig[sl] = True
            y_bin[sl] = np.isin(items, det)
            soft = np.isin(items, sure).astype(np.float32)
            if need > 0 and len(tied):
                soft += np.isin(items, tied) * (min(need, len(tied)) / len(tied))
            y_soft[sl] = soft
        out.update(y_cnt=y_cnt.astype(np.float32), y_bin=y_bin, y_soft=y_soft, elig=elig)
    return out


def get_D2(mode, refresh=False):
    """(sp, M, fb, D) with the v4 features cached on disk."""
    sp = Split(mode)
    M = m.Matrices(sp)
    fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
    path = PROC + f"rerank2_{mode}_{VERSION2}.npz"
    try:
        if refresh:
            raise FileNotFoundError
        z = np.load(path)
        D = {k: z[k] for k in z.files}
    except FileNotFoundError:
        allov = pd.read_parquet(PROC + "all_overlaps.parquet")
        D = build2(sp, M, allov)
        np.savez_compressed(path, **D)
    return sp, M, fb, D
