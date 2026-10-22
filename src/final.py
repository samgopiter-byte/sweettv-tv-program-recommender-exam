"""Final strategy: personal history x exposure + 0.25 * ALS (fixed config), evaluated on both folds and applied to test.

score(u, j) = minmax(personal_u(j) * exposure(j)) + 0.25 * minmax(ALS(u, j));  exactly 5 ids per user,
candidates = programmes airing in the target window, repeats allowed, cold-start users -> popularity fallback.
"""
import numpy as np
import pandas as pd
from common import Split, evaluate, ROOT, K
import methods as m

ALS_CFG = dict(factors=32, reg=150.0, alpha=1.0, iters=20, seed=42)
W_ALS = 0.25
GAMMA = 1.0
OUT = ROOT / "data" / "processed"


def final_scores(M):
    P_per, P_exp = m.personal_scores(M), m.exposure_scores(M)
    P_als = m.als_scores(M, **{k: v for k, v in ALS_CFG.items()})
    return m.rownorm(P_per * P_exp ** GAMMA, M.pool_ix) + W_ALS * m.rownorm(P_als, M.pool_ix)


if __name__ == "__main__":
    for mode in ("val", "val2"):
        sp = Split(mode)
        M = m.Matrices(sp)
        fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
        r = m.recommend(M, final_scores(M), fb)
        print(mode, "final strategy:", evaluate(r, sp.truth, sp.users))

    sp = Split("test")
    M = m.Matrices(sp)
    fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
    recs = m.recommend(M, final_scores(M), fb)
    sub = pd.read_csv(ROOT / "data" / "raw" / "submission.csv", dtype={"user_id": "uint64"})
    sub["tv_show_id"] = [" ".join(str(x) for x in recs[u]) for u in sub["user_id"]]
    (ROOT / "submissions").mkdir(exist_ok=True)
    sub.to_csv(ROOT / "submissions" / "submission_hybrid_v1.csv", index=False)
    pool = set(sp.pool.tolist())
    bad = [u for u in sub["user_id"] if len(set(recs[u])) != K or not set(recs[u]) <= pool]
    print("test: users", len(sub), "| train history users", int(M.has_hist[M.sel].sum()), "| pool", len(pool),
          "| lists not 5 unique pool ids:", len(bad), "| first rows:")
    print(sub.head(3).to_string(index=False))
