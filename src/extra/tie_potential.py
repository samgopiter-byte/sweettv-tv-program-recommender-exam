"""If the host broke ties by 'earliest first committed airing in the window', how much could a model that knows the rule gain?

Feature: delay (days) from the window start to the first airing of the candidate in the target window (schedule only).
Compared on the replica with the earliest-first truth (MAP_det) and with the random-tie expectation (MAP_exp).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import lightgbm as lgb

import lite
from common import PROC
import hs
from common import FOLDS

BASE = dict(learning_rate=0.05, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
            verbose=-1, seed=42, deterministic=True, force_row_wise=True, num_threads=4, objective="lambdarank", metric="ndcg",
            eval_at=[5], num_leaves=15, lambdarank_truncation_level=10)


def delay_feature(L, mo):
    from common import PROC
    sch = pd.read_parquet(os.path.join(PROC, "schedule.parquet"), columns=["tv_show_id", "start_s"])
    lo, hi = FOLDS[mo]
    sch = sch[(sch["tv_show_id"] != 0) & (sch["start_s"] >= lo) & (sch["start_s"] < hi)]
    first = sch.groupby("tv_show_id")["start_s"].min()
    items = L.meta[mo]["items"]
    d = np.full(len(items), 999.0)
    pos = pd.Series(np.arange(len(items)), index=items)
    ok = first.index.isin(items)
    d[pos.loc[first.index[ok]].to_numpy()] = (first[ok].to_numpy() - lo) / 86400.0
    return d[L.D[mo]["cols"]]


def crossfit(L, extra, folds=5):
    fo = L.folds_users(folds)
    idx = L.idx_all
    preds = {mo: np.zeros(len(L.D[mo]["rows"])) for mo in lite.FOLDS}
    for f in range(folds):
        Xs, ys, qs = [], [], []
        for k, mo in enumerate(lite.FOLDS):
            D = L.D[mo]
            m = (fo[D["rows"]] != f) & D["elig"]
            X = D["X"][m][:, idx]
            if extra is not None:
                X = np.column_stack([X, extra[mo][m]])
            Xs.append(X); ys.append(D["y_bin"][m].astype(np.int32)); qs.append(D["rows"][m] + k * L.n_users)
        X, y, q = np.vstack(Xs), np.concatenate(ys), np.concatenate(qs)
        _, counts = np.unique(q, return_counts=True)
        model = lgb.train(BASE, lgb.Dataset(X, y, group=counts), 200)
        for mo in lite.FOLDS:
            D = L.D[mo]
            te = fo[D["rows"]] == f
            Xt = D["X"][te][:, idx]
            if extra is not None:
                Xt = np.column_stack([Xt, extra[mo][te]])
            preds[mo][te] = model.predict(Xt)
    return preds


def score_det(L, mo, pred):
    me = L.meta[mo]
    lists = L.lists(mo, pred)
    r = hs.evaluate_samples(lists, me["samples"], me["eligible"], me["info"])
    return r["MAP@5"], r["MAP@5_det"]


if __name__ == "__main__":
    L = lite.Lite("v6")
    extra = {mo: delay_feature(L, mo) for mo in lite.FOLDS}
    for name, ex in (("baseline (no tie feature)", None), ("+ first-airing delay feature", extra)):
        P = crossfit(L, ex)
        res = {mo: score_det(L, mo, P[mo]) for mo in lite.FOLDS}
        print(f"{name:32s} expected-over-random-ties {np.mean([v[0] for v in res.values()]):.4f}   earliest-first truth {np.mean([v[1] for v in res.values()]):.4f}", flush=True)
