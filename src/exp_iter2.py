"""Where is the next gain? (1) learning curve: does more training data (users) help? (2) ranker hyper-parameters.

Scores: tie-aware Kaggle replica, mean over the five folds, 'all features + rank_bin' (attempt 2) cross-fit by users.
"""
import time
import numpy as np
import attempts2 as at

if __name__ == "__main__":
    ctx = at.Ctx(at.FOLDS_ALL)
    base = dict(at.KINDS["rank_bin"]["params"])

    def run(tag, folds=5):
        t = time.time()
        preds = ctx.crossfit("rank_bin", ctx.idx_all, folds=folds)
        sc = {mo: ctx.score(mo, preds[mo]) for mo in at.FOLDS_ALL}
        print(f"{tag:46s} val12 {sc['val12']:.4f}  mean {np.mean(list(sc.values())):.4f}  ({time.time() - t:.0f}s)", flush=True)
        return np.mean(list(sc.values()))

    print("== learning curve (share of users used for training) ==")
    for folds in (2, 3, 5, 10):
        run(f"{(folds - 1) / folds:.0%} of users", folds)

    print("== ranker hyper-parameters (80% of users) ==")
    res = []
    for leaves, rounds, md in [(7, 100, 100), (7, 200, 100), (15, 100, 100), (15, 200, 100), (31, 100, 100), (31, 200, 100),
                               (15, 100, 50), (15, 100, 200), (63, 100, 100)]:
        at.KINDS["rank_bin"]["params"] = dict(base, num_leaves=leaves, min_data_in_leaf=md)
        at.KINDS["rank_bin"]["rounds"] = rounds
        res.append((run(f"leaves {leaves} rounds {rounds} min_data {md}"), leaves, rounds, md))
    print("best:", sorted(res, reverse=True)[:3])
