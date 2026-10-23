"""Ensemble diversity: XGBoost and CatBoost rankers next to the LightGBM models (cross-fit by users, five folds, replica MAP@5)."""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker, Pool

import lite

BASE = dict(learning_rate=0.05, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
            verbose=-1, seed=42, deterministic=True, force_row_wise=True, num_threads=4)


def lab(kind, D):
    if kind in ("lgb_bin", "xgb_ndcg", "xgb_pair", "cat_yeti"):
        return D["y_bin"].astype(np.int32)
    if kind in ("lgb_soft", "xgb_soft"):
        return np.round(4 * D["y_soft"]).astype(np.int32)
    return D["y_cnt"]


def fit_predict(kind, X, y, qid, Xte):
    if kind == "lgb_bin" or kind == "lgb_soft":
        _, counts = np.unique(qid, return_counts=True)
        p = dict(BASE, objective="lambdarank", metric="ndcg", eval_at=[5], num_leaves=15, lambdarank_truncation_level=10)
        return lgb.train(p, lgb.Dataset(X, y, group=counts), 200).predict(Xte)
    if kind == "lgb_pois":
        return lgb.train(dict(BASE, objective="poisson", num_leaves=15), lgb.Dataset(X, y), 300).predict(Xte)
    if kind in ("xgb_ndcg", "xgb_pair", "xgb_soft"):
        obj = "rank:pairwise" if kind == "xgb_pair" else "rank:ndcg"
        mdl = xgb.XGBRanker(objective=obj, n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                            min_child_weight=5, reg_lambda=5.0, tree_method="hist", n_jobs=4, random_state=42)
        mdl.fit(X, y, qid=qid)
        return mdl.predict(Xte)
    if kind == "cat_yeti":
        mdl = CatBoostRanker(loss_function="YetiRank", iterations=300, depth=6, learning_rate=0.08, thread_count=4, verbose=0, random_seed=42)
        mdl.fit(Pool(X, y, group_id=qid))
        return mdl.predict(Xte)
    raise ValueError(kind)


def crossfit(L, kind, folds=5):
    fo = L.folds_users(folds)
    preds = {mo: np.zeros(len(L.D[mo]["rows"])) for mo in lite.FOLDS}
    idx = L.idx_all
    for f in range(folds):
        Xs, ys, qs = [], [], []
        for k, mo in enumerate(lite.FOLDS):
            D = L.D[mo]
            mtr = (fo[D["rows"]] != f) & D["elig"]
            Xs.append(D["X"][mtr][:, idx]); ys.append(lab(kind, D)[mtr]); qs.append(D["rows"][mtr] + k * L.n_users)
        X, y, q = np.vstack(Xs), np.concatenate(ys), np.concatenate(qs)
        # one model per fold of users, predict the held-out users of every temporal fold in one call
        te_parts = []
        for mo in lite.FOLDS:
            D = L.D[mo]
            te_parts.append(fo[D["rows"]] == f)
        Xte = np.vstack([L.D[mo]["X"][te][:, idx] for mo, te in zip(lite.FOLDS, te_parts)])
        pr = fit_predict(kind, X, y, q, Xte)
        pos = 0
        for mo, te in zip(lite.FOLDS, te_parts):
            n = int(te.sum())
            preds[mo][te] = pr[pos:pos + n]
            pos += n
    return preds


def zs(p, rows):
    s = pd.Series(p)
    g = s.groupby(rows)
    return ((s - g.transform("mean")) / (g.transform("std").fillna(1.0) + 1e-9)).to_numpy()


if __name__ == "__main__":
    L = lite.Lite("v6")
    kinds = sys.argv[1:] or ["lgb_bin", "lgb_soft", "lgb_pois", "xgb_ndcg", "xgb_pair", "cat_yeti"]
    P = {}
    for kind in kinds:
        t = time.time()
        P[kind] = crossfit(L, kind)
        sc = {mo: L.score(mo, P[kind][mo]) for mo in lite.FOLDS}
        print(f"{kind:10s} " + " ".join(f"{mo} {v:.4f}" for mo, v in sc.items()) + f"  mean {np.mean(list(sc.values())):.4f}  ({time.time() - t:.0f}s)", flush=True)
    import pickle
    pickle.dump(P, open(os.path.join(lite.SP, "div_preds.pkl"), "wb"))   # own cross-fit predictions, trusted local data
    def ens(names):
        sc = {mo: L.score(mo, sum(zs(P[k][mo], L.D[mo]["rows"]) for k in names)) for mo in lite.FOLDS}
        return sc, np.mean(list(sc.values()))
    combos = [["lgb_bin", "lgb_soft", "lgb_pois"], ["lgb_bin", "lgb_soft"], ["lgb_bin", "xgb_ndcg"], ["lgb_bin", "lgb_soft", "xgb_ndcg"],
              ["lgb_bin", "lgb_soft", "cat_yeti"], ["lgb_bin", "lgb_soft", "xgb_ndcg", "cat_yeti"], ["lgb_bin", "lgb_soft", "lgb_pois", "xgb_ndcg", "xgb_pair", "cat_yeti"]]
    for c in combos:
        if all(k in P for k in c):
            sc, mean = ens(c)
            print("ENS " + "+".join(c).ljust(52) + " ".join(f"{mo} {v:.4f}" for mo, v in sc.items()) + f"  mean {mean:.4f}", flush=True)
