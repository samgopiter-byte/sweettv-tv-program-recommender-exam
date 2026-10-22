"""Robustness to a change of window: train the re-ranker on one validation fold, apply it to the other.

User-disjoint (train users: folds 0-2, evaluation users: folds 3-4) and time-shifted (different history length
and target window). Compared with the in-fold 5-fold cross-fit restricted to the same evaluation users.
The two target windows overlap in time (2020-06-01..2020-06-28), so this is a diagnostic of feature shift,
not an unbiased accuracy estimate.
"""
import sys
import numpy as np
from common import evaluate
import rerank as rr

PARAMS = dict(num_leaves=15, min_data_in_leaf=100)
ROUNDS = 150

if __name__ == "__main__":
    rr.VERSION = sys.argv[1] if len(sys.argv) > 1 else rr.VERSION
    data = {mode: rr.get_D(mode) for mode in ("val", "val2")}
    n_users = len(data["val"][1].sel)
    fo = rr.user_folds(n_users)
    eval_users = fo >= 3
    print("feature version", rr.VERSION, "| features", len(data["val"][3]["names"]))
    for src, dst in (("val2", "val"), ("val", "val2")):
        sp, M, fb, Ds = data[dst]
        Dsrc = data[src][3]
        tr = fo[Dsrc["rows"]] < 3
        model = rr.fit(Dsrc["X"][tr], Dsrc["y"][tr], Dsrc["rows"][tr], ROUNDS, PARAMS)
        te = eval_users[Ds["rows"]]
        pred = model.predict(Ds["X"][te])
        users_te = [u for i, u in enumerate(sp.users) if eval_users[i]]
        lists = rr.lists_from_scores(sp, M, fb, Ds["rows"][te], Ds["cols"][te], pred)
        shifted = evaluate(lists, sp.truth, users_te)["MAP@5"]
        # in-fold reference: cross-fit inside dst, same evaluation users
        preds, _ = rr.crossfit_pooled([Ds], n_users, rounds=ROUNDS, params=PARAMS)
        lists_in = rr.lists_from_scores(sp, M, fb, Ds["rows"], Ds["cols"], preds[0])
        infold = evaluate(lists_in, sp.truth, users_te)["MAP@5"]
        bl = Ds["X"][:, list(Ds["names"]).index("blend")]
        base = evaluate(rr.lists_from_scores(sp, M, fb, Ds["rows"], Ds["cols"], bl), sp.truth, users_te)["MAP@5"]
        print(f"train on {src} -> evaluate on {dst} ({len(users_te)} users): shifted {shifted:.4f} | in-fold cross-fit {infold:.4f} | blend {base:.4f}")
