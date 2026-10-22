"""How much does each family of signals contribute to the final re-ranker?

Same protocol as final2.py (5-fold cross-fit by users, trained on both validation folds, same candidates),
only the feature set changes. 'studied methods' = the implemented ALS (MF), association rules, CBF and the
popularity baseline; the other families are repeat-consumption history and the time-of-week habit model.
"""
import numpy as np
from common import evaluate
import rerank as rr
from final2 import HIST_DEP, LGB, ROUNDS

REPEAT = ("per_", "per28_", "conv_")
FAMILIES = {
    "studied methods (ALS, AR, CBF) + popularity": lambda n: n.startswith(("als_", "ar_", "cbf_", "pop_")),
    "repeat history": lambda n: n.startswith(REPEAT) or n in ("reg_weeks", "days_since", "sec14", "sec14_rel", "airings_wk"),
    "time-of-week habit": lambda n: n.startswith("E_"),
    "context (airtime, category, user level)": lambda n: n in ("exp_rel", "n_air_wk", "airtime_wk", "n_channels", "med_dur",
                                                            "category", "u_sec_wk", "u_top_share", "u_cat_share"),
    "blend score": lambda n: n in ("blend", "blend_rank"),
}


def run(data, n_users, keep):
    names = list(data["val"][3]["names"])
    idx = [i for i, n in enumerate(names) if n not in HIST_DEP and keep(n)]
    Ds = [dict(data[m][3], X=data[m][3]["X"][:, idx]) for m in ("val", "val2")]
    preds, _ = rr.crossfit_pooled(Ds, n_users, rounds=ROUNDS, params=LGB)
    out = []
    for mode, D, p in zip(("val", "val2"), Ds, preds):
        sp, M, fb, _ = data[mode]
        out.append(evaluate(rr.lists_from_scores(sp, M, fb, D["rows"], D["cols"], p), sp.truth, sp.users)["MAP@5"])
    return out, len(idx)


if __name__ == "__main__":
    data = {m: rr.get_D(m) for m in ("val", "val2")}
    n_users = len(data["val"][1].sel)
    fam = list(FAMILIES)
    rows = []
    only = lambda keys: (lambda n: any(FAMILIES[k](n) for k in keys))
    experiments = [
        ("studied methods + popularity only", only([fam[0]])),
        ("studied methods + context", only([fam[0], fam[3]])),
        ("repeat history + context", only([fam[1], fam[3]])),
        ("habit + context", only([fam[2], fam[3]])),
        ("repeat + habit + context + blend (no studied methods)", only(fam[1:])),
        ("ALL features (final model)", lambda n: True),
    ]
    for k in fam:
        experiments.append((f"ALL minus {k}", (lambda kk: (lambda n: not FAMILIES[kk](n)))(k)))
    for name, keep in experiments:
        res, nf = run(data, n_users, keep)
        print(f"{name:62s} features {nf:2d}  MAP@5 val {res[0]:.4f}  val2 {res[1]:.4f}  mean {np.mean(res):.4f}", flush=True)
