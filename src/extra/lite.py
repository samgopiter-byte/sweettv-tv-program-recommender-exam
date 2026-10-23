"""Light evaluation context for experiments on the cached v6 features (no interaction tables kept in memory)."""
import os
import pickle
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

import hs
import methods as m
from common import PROC, Split, K, pad_to_k

SP = os.path.dirname(os.path.abspath(__file__))
FOLDS = ["val12", "val", "val2", "val3", "val5"]
HIST_DEP = ["mom14", "mom28", "log_user_frac", "span_frac"]


def build_meta(mode):
    # The pickles are written by this very function into the session scratchpad and read back only here (trusted local data).
    path = os.path.join(SP, f"meta_{mode}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    sp = Split(mode)
    M = m.Matrices(sp)
    fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
    info = hs.tie_info(hs.qualifying_counts(sp.inter, sp.tgt_lo, sp.tgt_hi))
    samples = hs.truth_samples(info)
    hist = hs.committed_matrix_all(sp, M)[M.sel]
    meta = dict(items=M.items, has=M.has_hist[M.sel], fb=fb, users=list(sp.users), samples=samples, info=info,
                eligible=[u for u in sp.users if u in samples], hist_cnt=np.asarray(hist.sum(axis=1)).ravel(),
                hist_items=[set(hist[i].indices.tolist()) for i in range(hist.shape[0])])
    pickle.dump(meta, open(path, "wb"))
    return meta


class Lite:
    def __init__(self, version="v6"):
        self.meta, self.D = {}, {}
        for mo in FOLDS:
            self.meta[mo] = build_meta(mo)
            z = np.load(PROC + f"rerank2_{mo}_{version}.npz")
            self.D[mo] = {k: z[k] for k in z.files}
        self.names = list(self.D[FOLDS[0]]["names"])
        self.idx_all = [i for i, n in enumerate(self.names) if n not in HIST_DEP]
        self.n_users = len(self.meta[FOLDS[0]]["users"])

    def lists(self, mo, pred):
        me, D = self.meta[mo], self.D[mo]
        df = pd.DataFrame({"r": D["rows"], "c": D["cols"], "p": pred}).sort_values(["r", "p"], ascending=[True, False])
        by = {r: g["c"].tolist() for r, g in df.groupby("r").head(K).groupby("r")}
        out = {}
        for i, u in enumerate(me["users"]):
            rec = [int(me["items"][j]) for j in by.get(i, [])] if me["has"][i] else []
            out[u] = pad_to_k(rec, me["fb"])
        return out

    def score(self, mo, pred):
        me = self.meta[mo]
        return hs.evaluate_samples(self.lists(mo, pred), me["samples"], me["eligible"], me["info"])["MAP@5"]

    def folds_users(self, folds=5, seed=0):
        return np.random.default_rng(seed).integers(0, folds, size=self.n_users)
