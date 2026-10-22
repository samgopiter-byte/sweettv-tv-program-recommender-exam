"""Where is the headroom? Recall of truth items within the top-k of the current best blend, and the oracle re-ranking bound."""
import numpy as np
from common import Split, evaluate
import methods as m
import habits as h

for mode in ("val", "val2"):
    sp = Split(mode); M = m.Matrices(sp)
    fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
    E = h.habit_scores(sp, M, half_life=56)
    P = m.personal_scores(M) * m.exposure_scores(M)
    S = m.rownorm(E, M.pool_ix) + 1.0 * m.rownorm(P, M.pool_ix)
    print(mode, "blend", evaluate(m.recommend(M, S, fb), sp.truth, sp.users))
    sub = S[:, M.pool_ix]
    order = np.argsort(-sub, axis=1)
    items_pool = M.items[M.pool_ix]
    rank_of = {}
    tot = {k: 0 for k in (5, 10, 20, 50, 100, 200)}
    n_truth = 0
    oracle = {k: [] for k in (5, 10, 20, 50)}
    for row, u in enumerate(sp.users):
        t = sp.truth.get(u)
        if not t or not M.has_hist[M.sel[row]]:
            continue
        ranked = items_pool[order[row, :200]]
        pos = {int(x): i for i, x in enumerate(ranked)}
        n_truth += len(t)
        for k in tot:
            tot[k] += sum(1 for x in t if pos.get(int(x), 10**9) < k)
        for k in oracle:
            hits = sum(1 for x in t if pos.get(int(x), 10**9) < k)
            oracle[k].append(min(hits, 5) / min(5, len(t)))
    print("  recall of truth items in top-k:", {k: round(v / n_truth, 3) for k, v in tot.items()})
    print("  oracle AP@5 if perfectly re-ranking the top-k candidates:", {k: round(float(np.mean(v)), 3) for k, v in oracle.items()})
