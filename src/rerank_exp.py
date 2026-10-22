"""Evaluate the re-ranker: baseline blend vs LambdaRank (single fold training vs pooled training), a few params."""
import itertools
import time
import numpy as np
from common import evaluate
import rerank as rr

if __name__ == "__main__":
    t = time.time()
    data = {}
    for mode in ("val", "val2"):
        data[mode] = rr.get_D(mode)
        sp, M, fb, D = data[mode]
        n_truth = sum(len(sp.truth[u]) for u in sp.users if u in sp.truth)
        print(f"{mode}: rows {len(D['rows'])}, positives {int(D['y'].sum())} of {n_truth} (candidate recall "
              f"{D['y'].sum() / n_truth:.3f}), features {len(D['names'])}  [{time.time() - t:.0f}s]", flush=True)
        bl = D["X"][:, list(D["names"]).index("blend")]
        base = rr.lists_from_scores(sp, M, fb, D["rows"], D["cols"], bl)
        print("  baseline blend:", round(evaluate(base, sp.truth, sp.users)["MAP@5"], 4))

    n_users = len(data["val"][1].sel)
    Ds = [data["val"][3], data["val2"][3]]
    grid = [dict(rounds=r, params=dict(num_leaves=nl, min_data_in_leaf=md))
            for r, nl, md in itertools.product((60, 100, 150, 250), (7, 15), (40, 100))]
    out = []
    for g in grid:
        for pooled in (False, True):
            if pooled:
                preds, _ = rr.crossfit_pooled(Ds, n_users, rounds=g["rounds"], params=g["params"])
            else:
                preds = []
                for D in Ds:
                    p, _ = rr.crossfit_pooled([D], n_users, rounds=g["rounds"], params=g["params"])
                    preds.append(p[0])
            res = []
            for mode, D, p in zip(("val", "val2"), Ds, preds):
                sp, M, fb, _ = data[mode]
                res.append(evaluate(rr.lists_from_scores(sp, M, fb, D["rows"], D["cols"], p), sp.truth, sp.users)["MAP@5"])
            out.append((np.mean(res), res, g, pooled))
            print(f"rounds {g['rounds']:3d} leaves {g['params']['num_leaves']:2d} min_data {g['params']['min_data_in_leaf']:3d} "
                  f"pooled={pooled!s:5s} -> val {res[0]:.4f} val2 {res[1]:.4f} mean {np.mean(res):.4f}", flush=True)
    out.sort(key=lambda x: -x[0])
    print("best:", [(round(a, 4), [round(x, 4) for x in b], c, d) for a, b, c, d in out[:4]])
