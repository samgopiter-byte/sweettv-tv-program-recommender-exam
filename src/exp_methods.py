"""The four methods of the course (MF=ALS, association rules, CBF, PPR) and popularity, each evaluated individually with the
Kaggle replica on the same folds / pool / protocol, then two hybrids by rank fusion (RRF) for comparison with the learned
re-ranker. Every method is given two inputs: the old 'watched' seconds matrix and the truth-aligned committed-airings matrix.
"""
import copy
import sys
import time
import numpy as np
import scipy.sparse as sps

import hs
import methods as m
import rerank as rr
from common import Split

FOLDS = ["val12", "val", "val2", "val3", "val5"]


def rrf(mats, pool_ix, k=60.0):
    out = 0
    for S in mats:
        out = out + 1.0 / (k + rr.rank_matrix(S, pool_ix))
    return out


def run(mode, rows):
    sp = Split(mode)
    M = m.Matrices(sp)
    fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
    info = hs.tie_info(hs.qualifying_counts(sp.inter, sp.tgt_lo, sp.tgt_hi))
    samples = hs.truth_samples(info)
    users = [u for u in sp.users if u in samples]
    import pandas as pd
    allov = pd.read_parquet(rr.PROC + "all_overlaps.parquet")
    X = hs.committed_matrix_all(sp, M)
    Mc = m.with_input(M, X)
    sc = {}
    sc["popularity (watched seconds)"] = m.pop_scores(M)
    sc["MF: ALS, watched-seconds input"] = m.als_scores(M, **rr.ALS_CFG)
    sc["MF: ALS, committed input"] = m.als_scores(Mc, factors=32, reg=50.0, alpha=5.0)
    sc["AR: rules, watched basket"] = m.assoc_scores(M, min_supp=10, beta=0.0)
    sc["AR: rules, committed basket"] = m.assoc_scores(Mc, min_supp=10, beta=0.3)
    sc["CBF: tf-idf, watched profile"] = m.cbf_scores(M)
    sc["CBF: tf-idf, committed profile"] = m.cbf_scores(Mc)
    for d in (0.5, 0.85):
        sc[f"PPR d={d}, committed graph"] = m.ppr_scores(M, X, d=d)
    hab = hs.habit_hs(sp, M, allov, half_life=56, pooled=True)
    conv = hs.conversion_hs(sp, M, hs.hs_matrices(sp, M, halflives=(None,))["hs_cnt"])
    sc["(extra) committed conversion"] = conv
    sc["(extra) slot habit, committed"] = hab
    course = ["MF: ALS, committed input", "AR: rules, committed basket", "CBF: tf-idf, committed profile", "PPR d=0.85, committed graph"]
    sc["HYBRID rank fusion: ALS + AR + CBF + PPR"] = rrf([sc[k] for k in course], M.pool_ix)
    sc["HYBRID rank fusion: course methods + habit + conversion"] = rrf([sc[k] for k in course] + [hab, conv], M.pool_ix)
    for name, S in sc.items():
        lists = m.recommend(M, S, fb)
        r = hs.evaluate_samples(lists, samples, users, info)
        rows.setdefault(name, {})[mode] = r["MAP@5"]
    print(f"[{mode}] done, eligible users {len(users)}", flush=True)


if __name__ == "__main__":
    modes = sys.argv[1:] or FOLDS
    rows = {}
    for mode in modes:
        t = time.time()
        run(mode, rows)
    print(f"{'method (expected MAP@5, Kaggle replica)':62s}" + "".join(f"{mo:>8s}" for mo in modes) + "    mean")
    for name, d in rows.items():
        print(f"{name:62s}" + "".join(f"{d[mo]:8.4f}" for mo in modes) + f"  {np.mean(list(d.values())):7.4f}")
