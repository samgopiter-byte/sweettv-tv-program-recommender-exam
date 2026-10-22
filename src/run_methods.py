"""Validation comparison: popularity, personal-history, ALS, association rules, CBF, and hybrids.

Everything uses the same split (train < 2020-06-01, target 2020-06-01..2020-07-26), the same candidate
pool and the same evaluator. Hyper-parameters are chosen on this validation window only.
"""
import pickle
import time
import numpy as np
import pandas as pd
from common import Split, evaluate, ROOT
import methods as m

OUT = ROOT / "data" / "processed"

if __name__ == "__main__":
    sp = Split("val")
    M = m.Matrices(sp)
    fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
    users = sp.users
    print("items", len(M.items), "pool", len(M.pool_ix), "users scored", len(users),
          "with history", int(M.has_hist[M.sel].sum()))

    rows, recs, scores = [], {}, {}

    def run(name, S, repeats=True, keep=False):
        r = m.recommend(M, S, fb, repeats)
        ev = evaluate(r, sp.truth, users)
        ev["method"] = name
        rows.append(ev)
        recs[name] = r
        if keep:
            scores[name] = S
        print(f"{name:44s} MAP@5 {ev['MAP@5']:.4f}  Hit@5 {ev['HitRate@5']:.3f}", flush=True)

    t = time.time(); P_pop = m.pop_scores(M); run("popularity (train seconds)", P_pop, keep=True)
    P_per = m.personal_scores(M); run("personal history (own seconds)", P_per, keep=True)

    best = (-1, None)
    for f in (16, 32, 64):
        for reg in (1.0, 10.0, 50.0):
            for alpha in (1.0, 10.0):
                S = m.als_scores(M, factors=f, reg=reg, alpha=alpha)
                name = f"als f{f} reg{reg:g} a{alpha:g}"
                run(name, S)
                if rows[-1]["MAP@5"] > best[0]:
                    best = (rows[-1]["MAP@5"], (f, reg, alpha))
    print("best ALS", best)
    f, reg, alpha = best[1]
    P_als = m.als_scores(M, factors=f, reg=reg, alpha=alpha); run("ALS (best)", P_als, keep=True)

    bestA = (-1, None)
    for ms in (10, 30):
        for beta in (0.0, 0.3, 0.6, 1.0):
            S = m.assoc_scores(M, min_supp=ms, beta=beta)
            name = f"assoc supp>={ms} beta{beta:g}"
            run(name, S)
            if rows[-1]["MAP@5"] > bestA[0]:
                bestA = (rows[-1]["MAP@5"], (ms, beta))
    print("best assoc", bestA)
    P_ar = m.assoc_scores(M, min_supp=bestA[1][0], beta=bestA[1][1]); run("Association rules (best)", P_ar, keep=True)

    P_cbf = m.cbf_scores(M); run("CBF (tf-idf attributes)", P_cbf, keep=True)

    # hybrids: personal history + CF signal (+ popularity prior); weights on a coarse grid
    bestH = (-1, None)
    for wa in (0.0, 0.25, 0.5, 1.0):
        for wr in (0.0, 0.25, 0.5, 1.0):
            for wp in (0.0, 0.25, 0.5):
                if wa == wr == wp == 0:
                    continue
                S = m.blend(M, [(1.0, P_per), (wa, P_als), (wr, P_ar), (wp, P_pop)])
                r = m.recommend(M, S, fb)
                ev = evaluate(r, sp.truth, users)
                if ev["MAP@5"] > bestH[0]:
                    bestH = (ev["MAP@5"], (wa, wr, wp), ev)
    print("best hybrid (personal=1.0, als, assoc, pop):", bestH[1], bestH[2])
    wa, wr, wp = bestH[1]
    S = m.blend(M, [(1.0, P_per), (wa, P_als), (wr, P_ar), (wp, P_pop)])
    run(f"hybrid personal+als{wa:g}+assoc{wr:g}+pop{wp:g}", S, keep=True)
    run("hybrid (no repeats)", S, repeats=False)
    run("ALS best (no repeats)", P_als, repeats=False)
    run("assoc best (no repeats)", P_ar, repeats=False)
    run("CBF (no repeats)", P_cbf, repeats=False)
    run("popularity (no repeats)", P_pop, repeats=False)

    df = pd.DataFrame(rows)[["method", "users", "MAP@5", "HitRate@5"]]
    df.to_csv(OUT / "val_results.csv", index=False)
    with open(OUT / "val_recs.pkl", "wb") as fh:
        pickle.dump(recs, fh)
    with open(OUT / "val_best_params.txt", "w") as fh:
        fh.write(f"als={best[1]} assoc={bestA[1]} hybrid_weights(als,assoc,pop)={bestH[1]}\n")
    print(df.tail(12).to_string(index=False))
