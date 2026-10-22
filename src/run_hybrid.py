"""Two validation folds: individual methods, then personal-history hybrids with an exposure term."""
import itertools, json, pickle
import numpy as np
from common import Split, evaluate, ROOT
import methods as m

OUT = ROOT / "data" / "processed"
ALS_GRID = [(32, 50.0, 1.0), (32, 150.0, 1.0), (32, 400.0, 1.0), (32, 150.0, 0.3), (16, 150.0, 1.0), (64, 150.0, 1.0)]


def fold(mode):
    sp = Split(mode)
    M = m.Matrices(sp)
    fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
    users = sp.users
    ev = lambda S: evaluate(m.recommend(M, S, fb), sp.truth, users)
    res = {"individual": {}}
    P = {"pop": m.pop_scores(M), "per": m.personal_scores(M), "exp": m.exposure_scores(M)}
    res["individual"]["popularity"] = ev(P["pop"])
    res["individual"]["personal"] = ev(P["per"])
    res["individual"]["exposure only"] = ev(P["exp"])
    bestA = None
    res["als_grid"] = {}
    Pals = {}
    for cfg in ALS_GRID:
        S = m.als_scores(M, factors=cfg[0], reg=cfg[1], alpha=cfg[2])
        Pals[cfg] = S
        r = ev(S)
        res["als_grid"][str(cfg)] = r["MAP@5"]
        if bestA is None or r["MAP@5"] > bestA[0]:
            bestA = (r["MAP@5"], cfg)
    res["individual"]["ALS best " + str(bestA[1])] = ev(Pals[bestA[1]])
    P["als"] = Pals[bestA[1]]
    P["ar"] = m.assoc_scores(M, min_supp=10, beta=0.0)
    res["individual"]["association rules (supp>=10, conf)"] = ev(P["ar"])
    P["cbf"] = m.cbf_scores(M)
    res["individual"]["CBF"] = ev(P["cbf"])
    N = {k: m.rownorm(v, M.pool_ix) for k, v in P.items() if k != "exp"}
    res["bestALS"] = bestA[1]
    hy = {}
    for g in (0.0, 0.25, 0.5, 1.0):
        Ng = m.rownorm(P["per"] * P["exp"] ** g, M.pool_ix)
        for wa, wr, wp in itertools.product((0, 0.25, 0.5, 1.0), (0, 0.25, 0.5, 1.0), (0, 0.25, 0.5)):
            S = Ng + wa * N["als"] + wr * N["ar"] + wp * N["pop"]
            hy[(g, wa, wr, wp)] = ev(S)["MAP@5"]
    return res, hy, (sp, M, P, N, fb)


if __name__ == "__main__":
    out = {}
    hys = {}
    for mode in ("val", "val2"):
        res, hy, _ = fold(mode)
        out[mode] = res
        hys[mode] = hy
        print("==", mode, "==")
        for k, v in res["individual"].items():
            print(f"  {k:44s} MAP@5 {v['MAP@5']:.4f} Hit@5 {v['HitRate@5']:.3f}")
        print("  ALS grid:", res["als_grid"])
    both = {k: (hys["val"][k], hys["val2"][k]) for k in hys["val"]}
    top = sorted(both.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:8]
    print("top hybrid configs (g, w_als, w_assoc, w_pop): (MAP val, MAP val2)")
    for k, v in top:
        print("  ", k, [round(x, 4) for x in v])
    print("reference g=0, only personal:", [round(x, 4) for x in both[(0.0, 0, 0, 0)]] if (0.0, 0, 0, 0) in both else "n/a")
    with open(OUT / "hybrid_grid.pkl", "wb") as fh:
        pickle.dump({"both": both, "out": out}, fh)
