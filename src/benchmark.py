"""Wall-clock time of every method on the 12-week fold (single run, CPU only; the machine is described in the README).

Output: results/runtime.md
"""
import time
import platform
import numpy as np
import pandas as pd

import hs
import methods as m
import rerank as rr
from common import ROOT, Split

if __name__ == "__main__":
    rows = []

    def timed(name, fn):
        t = time.time()
        out = fn()
        rows.append((name, time.time() - t))
        print(f"{name:55s} {rows[-1][1]:7.1f} s", flush=True)
        return out

    sp = timed("load fold (interactions, schedule, history table)", lambda: Split("val12"))
    M = timed("index maps and user x item matrices", lambda: m.Matrices(sp))
    allov = pd.read_parquet(rr.PROC + "all_overlaps.parquet")
    X = timed("committed-airing matrix of all users", lambda: hs.committed_matrix_all(sp, M))
    Mc = m.with_input(M, X)
    fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
    timed("popularity (baseline)", lambda: m.pop_scores(M))
    timed("MF: ALS, 32 factors, 20 iterations", lambda: m.als_scores(Mc, factors=32, reg=50.0, alpha=5.0))
    timed("AR: association rules, pairwise support/confidence", lambda: m.assoc_scores(Mc, min_supp=10, beta=0.3))
    timed("CBF: tf-idf attribute profiles", lambda: m.cbf_scores(Mc))
    timed("PPR: co-view graph + 15 power iterations (d = 0.85)", lambda: m.ppr_scores(M, X, d=0.85))
    timed("slot habit (committed), pooled profile", lambda: hs.habit_hs(sp, M, allov, half_life=56, pooled=True))
    timed("committed conversion", lambda: hs.conversion_hs(sp, M, hs.hs_matrices(sp, M, halflives=(None,))["hs_cnt"]))
    import rerank2 as r2
    timed("all features of one fold (13 scorers, candidates, 104 features)", lambda: r2.build2(sp, M, allov))
    lines = ["# Runtime on the 12-week fold (one run, CPU only)\n",
             f"Machine: {platform.processor()}, {platform.system()} {platform.release()}, Python {platform.python_version()}; threads: BLAS default, LightGBM 4.\n",
             "| step | seconds |", "|---|---|"] + [f"| {n} | {s:.1f} |" for n, s in rows]
    (ROOT / "results" / "runtime.md").write_text("\n".join(lines), encoding="utf-8")
