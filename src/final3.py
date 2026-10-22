"""Final model v3: train the chosen attempt on all folds / all eligible users and predict the test window.

    python final3.py A2|A3|A4|A5 [output name]

A2 = LambdaRank (deterministic top-5 labels), A3 = Poisson regression of committed-airing rate, A4 = graded LambdaRank on
tie-aware soft labels, A5 = per-user z-scored average of A2, A3 and A4. All use the full truth-aligned feature set.
"""
import sys
import numpy as np
import pandas as pd

import rerank as rr
import rerank2 as r2
import attempts2 as at
from common import ROOT, K
from final2 import write_submission

SPEC = {"A2": [("rank_bin", "all")], "A3": [("poisson", "all")], "A4": [("rank_soft", "all")],
        "A5": [("rank_bin", "all"), ("poisson", "all"), ("rank_soft", "all")], "A1": [("rank_bin", "old")]}


def fit_full(ctx, kind, idx):
    Xs, ys, rs = [], [], []
    for k, mo in enumerate(ctx.folds):
        D = ctx.data[mo][3]
        mtr = D["elig"]
        Xs.append(D["X"][mtr][:, idx]); ys.append(at.labels(kind, D)[mtr]); rs.append(D["rows"][mtr] + k * ctx.n_users)
    return at.train(kind, np.vstack(Xs), np.concatenate(ys), np.concatenate(rs))


if __name__ == "__main__":
    which = sys.argv[1]
    out_name = sys.argv[2] if len(sys.argv) > 2 else f"submission_{which}.csv"
    ctx = at.Ctx(at.FOLDS_ALL)
    spT, MT, fbT, DT = r2.get_D2("test")
    assert list(DT["names"]) == ctx.names, "feature layout differs between folds and test"
    parts = []
    for kind, fs in SPEC[which]:
        idx = ctx.idx_all if fs == "all" else ctx.idx_old
        model = fit_full(ctx, kind, idx)
        parts.append(at.zscore(model.predict(DT["X"][:, idx]), DT["rows"]) if len(SPEC[which]) > 1 else model.predict(DT["X"][:, idx]))
        imp = sorted(zip(model.feature_importance("gain"), [ctx.names[i] for i in idx]), reverse=True)
        tot = sum(x for x, _ in imp)
        print(kind, "top features by gain:", [(n, round(float(g) / tot, 3)) for g, n in imp[:8]])
    pred = sum(parts)
    lists = rr.lists_from_scores(spT, MT, fbT, DT["rows"], DT["cols"], pred)
    sub = write_submission(spT, lists, ROOT / "submissions" / out_name)
    pool = set(spT.pool.tolist())
    bad = [u for u in spT.users if len(set(lists[u])) != K or not set(lists[u]) <= pool]
    print(f"wrote submissions/{out_name}: {len(sub)} rows, lists that are not 5 unique pool ids: {len(bad)}")
