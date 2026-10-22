"""Truth-aligned scorers alone, evaluated with the tie-aware Kaggle replica (template users with >= 5 committed programmes)."""
import time
import numpy as np
import pandas as pd

from common import Split, test_users
import methods as m
import hs
import rerank as rr


def run(mode):
    sp = Split(mode)
    M = m.Matrices(sp)
    fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
    cnt = hs.qualifying_counts(sp.inter, sp.tgt_lo, sp.tgt_hi)
    info = hs.tie_info(cnt)
    samples = hs.truth_samples(info)
    users = [u for u in sp.users if u in samples]
    allov = pd.read_parquet(rr.PROC + "all_overlaps.parquet")
    t = time.time()
    B = hs.hs_scores(sp, M, allov)
    print(f"[{mode}] hs scorers computed in {time.time() - t:.0f}s; eligible template users {len(users)}", flush=True)
    old = {"pop": m.pop_scores(M), "per": m.personal_scores(M), "exp": m.exposure_scores(M), "E_f56": h_old(sp, M)}
    old["per*exp"] = old["per"] * old["exp"]
    res = {}
    for name, S in {**{k: v for k, v in old.items() if k in ("pop", "per*exp", "E_f56")}, **B}.items():
        lists = m.recommend(M, S, fb)
        r = hs.evaluate_samples(lists, samples, users, info)
        res[name] = r
        print(f"  {name:12s} MAP@5 (expected over tie-breaks) {r['MAP@5']:.4f}   deterministic earliest-first {r['MAP@5_det']:.4f}", flush=True)
    return res


def h_old(sp, M):
    import habits as hb
    return hb.habit_scores(sp, M, half_life=56)


if __name__ == "__main__":
    import sys
    for mode in sys.argv[1:] or ["val12", "val"]:
        run(mode)
