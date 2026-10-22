"""Does the habit model help? Individual scores and simple blends on both validation folds."""
import itertools
import sys
import time
import numpy as np
from common import Split, evaluate
import methods as m
import habits as h


def conversion_scores(M, c=3600.0):
    """E2[u, j] = user's training seconds on j / airtime of j in training  x  airtime of j in the target window."""
    sp = M.sp
    s = sp.sched[sp.sched["tv_show_id"] != 0]
    tr = s[(s["start_s"] >= sp.train_lo) & (s["start_s"] < sp.train_hi)].groupby("tv_show_id")["duration"].sum()
    tg = s[(s["start_s"] >= sp.tgt_lo) & (s["start_s"] < sp.tgt_hi)].groupby("tv_show_id")["duration"].sum()
    t_tr = np.zeros(len(M.items)); t_tg = np.zeros(len(M.items))
    for it, x in tr.items():
        if it in M.item_ix: t_tr[M.item_ix[it]] = x
    for it, x in tg.items():
        if it in M.item_ix: t_tg[M.item_ix[it]] = x
    S = M.sec[M.sel].toarray()
    return S / (t_tr[None, :] + c) * t_tg[None, :]


def fold(mode):
    sp = Split(mode)
    M = m.Matrices(sp)
    fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
    ev = lambda S: evaluate(m.recommend(M, S, fb), sp.truth, sp.users)["MAP@5"]
    B = {}
    B["pop"] = m.pop_scores(M)
    B["per"] = m.personal_scores(M)
    B["exp"] = m.exposure_scores(M)
    B["per*exp"] = B["per"] * B["exp"]
    B["conv"] = conversion_scores(M)
    for name, kw in [("E fine", {}), ("E fine h28", {"half_life": 28}), ("E fine h56", {"half_life": 56}),
                     ("E pooled", {"pooled": True}), ("E pooled h28", {"pooled": True, "half_life": 28}),
                     ("E pooled h56", {"pooled": True, "half_life": 56})]:
        t = time.time()
        B[name] = h.habit_scores(sp, M, **kw)
        print(f"  [{mode}] computed {name} in {time.time() - t:.0f}s", flush=True)
    res = {k: ev(v) for k, v in B.items()}
    return sp, M, fb, B, res


if __name__ == "__main__":
    allres = {}
    for mode in ("val", "val2"):
        sp, M, fb, B, res = fold(mode)
        allres[mode] = res
        print("==", mode)
        for k, v in res.items():
            print(f"  {k:14s} MAP@5 {v:.4f}")
        # blends: habit (best fine/pooled) + per*exp + conv, row-normalised
        N = {k: m.rownorm(v, M.pool_ix) for k, v in B.items()}
        ev = lambda S: evaluate(m.recommend(M, S, fb), sp.truth, sp.users)["MAP@5"]
        best = []
        for e1, e2 in itertools.product(("E fine", "E fine h56", "E pooled", "E pooled h56"), ("per*exp", "conv", "per")):
            for w in (0.25, 0.5, 1.0, 2.0):
                best.append((ev(N[e1] + w * N[e2]), e1, e2, w))
        best.sort(reverse=True)
        print("  top blends (habit + w*other):", [(round(a, 4), b, c, d) for a, b, c, d in best[:6]])
