"""Five attempts, scored with the tie-aware Kaggle replica (expected MAP@5 over random tie-breaks, template users that
have a truth), on five temporal folds. Two protocols:
  cross-fit : 5-fold by users, models trained on the other users of all five folds
  transfer  : train on the other four folds using half of the users, evaluate on the held-out fold's other half
              (user-disjoint and a fold with an unseen cutoff / horizon), halves swapped
Attempts
  1. old v3 features, LambdaRank on the replica's deterministic top-5 labels   (effect of the aligned labels only)
  2. all features (old + committed-airing scorers), LambdaRank, deterministic labels
  3. all features, Poisson regression of the committed-airing rate per week
  4. all features, graded LambdaRank on tie-aware soft labels
  5. ensemble of 2, 3 and 4 (per-user z-scored scores averaged)
"""
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

import os

import hs
import methods as m
import rerank as rr
import rerank2 as r2

if os.environ.get("R2_VERSION"):                 # e.g. R2_VERSION=v5 to evaluate an older feature set
    r2.VERSION2 = os.environ["R2_VERSION"]

FOLDS_ALL = ["val12", "val", "val2", "val3", "val5"]
HIST_DEP = ["mom14", "mom28", "log_user_frac", "span_frac"]
BASE = dict(learning_rate=0.05, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=5.0, verbose=-1, seed=42, deterministic=True, force_row_wise=True, num_threads=4)
KINDS = {
    # 15 leaves / 200 rounds chosen in exp_iter2.py (mean replica MAP@5 0.4114 vs 0.4064 for 7 leaves / 100 rounds)
    "rank_bin": dict(params=dict(BASE, objective="lambdarank", metric="ndcg", eval_at=[5], num_leaves=15, lambdarank_truncation_level=10), rounds=200),
    "rank_soft": dict(params=dict(BASE, objective="lambdarank", metric="ndcg", eval_at=[5], num_leaves=15, lambdarank_truncation_level=10), rounds=200),
    "poisson": dict(params=dict(BASE, objective="poisson", num_leaves=15), rounds=300),
    "xent": dict(params=dict(BASE, objective="cross_entropy", num_leaves=15), rounds=300),
}


def labels(kind, D):
    if kind == "rank_bin":
        return D["y_bin"].astype(np.int32)
    if kind == "rank_soft":
        return np.round(4 * D["y_soft"]).astype(np.int32)
    if kind == "poisson":
        return D["y_cnt"]
    return D["y_soft"]


def train(kind, X, y, rows):
    cfg = KINDS[kind]
    if kind.startswith("rank"):
        order = np.argsort(rows, kind="stable")
        _, counts = np.unique(rows[order], return_counts=True)
        ds = lgb.Dataset(X[order], y[order], group=counts)
    else:
        ds = lgb.Dataset(X, y)
    return lgb.train(cfg["params"], ds, num_boost_round=cfg["rounds"])


class Ctx:
    def __init__(self, folds):
        self.folds = folds
        self.data = {mo: r2.get_D2(mo) for mo in folds}
        self.ev = {}
        for mo in folds:
            sp, M, fb, D = self.data[mo]
            info = hs.tie_info(hs.qualifying_counts(sp.inter, sp.tgt_lo, sp.tgt_hi))
            samples = hs.truth_samples(info)
            self.ev[mo] = dict(info=info, samples=samples, users=[u for u in sp.users if u in samples])
        names = list(self.data[folds[0]][3]["names"])
        self.names = names
        self.idx_all = [i for i, n in enumerate(names) if n not in HIST_DEP]
        self.idx_old = [i for i in self.idx_all if not names[i].startswith(("hs_", "hab_hs")) and names[i] != "n_hist_air_wk"]
        self.n_users = len(self.data[folds[0]][1].sel)

    def score(self, mode, pred, rows=None, cols=None, users=None):
        sp, M, fb, D = self.data[mode]
        r = D["rows"] if rows is None else rows
        c = D["cols"] if cols is None else cols
        lists = rr.lists_from_scores(sp, M, fb, r, c, pred)
        e = self.ev[mode]
        return hs.evaluate_samples(lists, e["samples"], e["users"] if users is None else users, e["info"])["MAP@5"]

    def crossfit(self, kind, idx, folds=5, seed=0):
        fo = rr.user_folds(self.n_users, folds, seed)
        preds = {mo: np.zeros(len(self.data[mo][3]["rows"])) for mo in self.folds}
        for f in range(folds):
            Xs, ys, rs = [], [], []
            for k, mo in enumerate(self.folds):
                D = self.data[mo][3]
                mtr = (fo[D["rows"]] != f) & D["elig"]
                Xs.append(D["X"][mtr][:, idx]); ys.append(labels(kind, D)[mtr]); rs.append(D["rows"][mtr] + k * self.n_users)
            model = train(kind, np.vstack(Xs), np.concatenate(ys), np.concatenate(rs))
            for mo in self.folds:
                D = self.data[mo][3]
                te = fo[D["rows"]] == f
                preds[mo][te] = model.predict(D["X"][te][:, idx])
        return preds

    def transfer(self, kind, idx, eval_modes, seed=0):
        """Train on the other folds (half of the users), predict the held-out fold's other half; halves swapped."""
        fo = rr.user_folds(self.n_users, 2, seed)
        out = {}
        for ev_mode in eval_modes:
            pred = np.zeros(len(self.data[ev_mode][3]["rows"]))
            for half in (0, 1):
                Xs, ys, rs = [], [], []
                for k, mo in enumerate(self.folds):
                    if mo == ev_mode:
                        continue
                    D = self.data[mo][3]
                    mtr = (fo[D["rows"]] != half) & D["elig"]
                    Xs.append(D["X"][mtr][:, idx]); ys.append(labels(kind, D)[mtr]); rs.append(D["rows"][mtr] + k * self.n_users)
                model = train(kind, np.vstack(Xs), np.concatenate(ys), np.concatenate(rs))
                De = self.data[ev_mode][3]
                te = fo[De["rows"]] == half
                pred[te] = model.predict(De["X"][te][:, idx])
            out[ev_mode] = pred
        return out


def zscore(pred, rows):
    s = pd.Series(pred)
    g = s.groupby(rows)
    return ((s - g.transform("mean")) / (g.transform("std").fillna(1.0) + 1e-9)).to_numpy()


if __name__ == "__main__":
    t0 = time.time()
    ctx = Ctx(FOLDS_ALL)
    print(f"data loaded in {time.time() - t0:.0f}s; features {len(ctx.names)} (old {len(ctx.idx_old)}, used {len(ctx.idx_all)}); users {ctx.n_users}", flush=True)
    results = {}

    # reference scorers without learning
    for name in ("pop_rel", "hab_hs_p_rel", "hab_hs_rel", "hs_conv_rel", "hs_cnt56_rel"):
        j = ctx.names.index(name)
        sc = {mo: ctx.score(mo, ctx.data[mo][3]["X"][:, j]) for mo in FOLDS_ALL}
        results["ref " + name] = sc
        print(f"{'ref ' + name:36s}", {k: round(v, 4) for k, v in sc.items()}, "mean", round(float(np.mean(list(sc.values()))), 4), flush=True)

    attempts = [("A1 old features + rank_bin", "rank_bin", ctx.idx_old), ("A2 all features + rank_bin", "rank_bin", ctx.idx_all),
                ("A3 all features + poisson", "poisson", ctx.idx_all), ("A4 all features + rank_soft", "rank_soft", ctx.idx_all)]
    cf = {}
    for name, kind, idx in attempts:
        t = time.time()
        preds = ctx.crossfit(kind, idx)
        cf[name] = preds
        sc = {mo: ctx.score(mo, preds[mo]) for mo in FOLDS_ALL}
        results[name] = sc
        print(f"{name:36s}", {k: round(v, 4) for k, v in sc.items()}, "mean", round(float(np.mean(list(sc.values()))), 4), f"({time.time() - t:.0f}s)", flush=True)
    # A5 ensemble
    ens = {mo: sum(zscore(cf[n][mo], ctx.data[mo][3]["rows"]) for n in ("A2 all features + rank_bin", "A3 all features + poisson", "A4 all features + rank_soft")) for mo in FOLDS_ALL}
    sc = {mo: ctx.score(mo, ens[mo]) for mo in FOLDS_ALL}
    results["A5 ensemble of A2+A3+A4"] = sc
    print(f"{'A5 ensemble of A2+A3+A4':36s}", {k: round(v, 4) for k, v in sc.items()}, "mean", round(float(np.mean(list(sc.values()))), 4), flush=True)

    print("\n== transfer protocol (unseen fold, user-disjoint), evaluation folds val12, val, val2 ==")
    ev_modes = ["val12", "val", "val2"]
    tr = {}
    for name, kind, idx in attempts:
        p = ctx.transfer(kind, idx, ev_modes)
        tr[name] = p
        sc = {mo: ctx.score(mo, p[mo]) for mo in ev_modes}
        print(f"{name:36s}", {k: round(v, 4) for k, v in sc.items()}, "mean", round(float(np.mean(list(sc.values()))), 4), flush=True)
    ens = {mo: sum(zscore(tr[n][mo], ctx.data[mo][3]["rows"]) for n in ("A2 all features + rank_bin", "A3 all features + poisson", "A4 all features + rank_soft")) for mo in ev_modes}
    sc = {mo: ctx.score(mo, ens[mo]) for mo in ev_modes}
    print(f"{'A5 ensemble of A2+A3+A4':36s}", {k: round(v, 4) for k, v in sc.items()}, "mean", round(float(np.mean(list(sc.values()))), 4), flush=True)
