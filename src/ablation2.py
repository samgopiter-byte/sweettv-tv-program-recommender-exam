"""Contribution of each family of signals to the final re-ranker, scored with the Kaggle replica.

Model: LambdaRank (15 leaves, 200 rounds) on the replica's deterministic labels, 5-fold cross-fit by users, five temporal
folds. Only the feature set changes. Families follow the story of the report: the four course methods (MF/ALS, association
rules, CBF, PPR) fed with watched seconds or with committed airings, popularity, committed-airing counts, the slot habit,
the older seconds-based history/habit features, and context.
"""
import numpy as np
import attempts2 as at

FAMILIES = {
    "course methods (ALS, AR, CBF, PPR)": lambda n: n.startswith(("als_", "ar_", "cbf_", "ppr")),
    "popularity (seconds and committed)": lambda n: n.startswith(("pop_",)),
    "committed counts / conversion": lambda n: n.startswith(("hs_", "n_hist_air")) and not n.startswith("hs_conv_dummy"),
    "slot habit (committed)": lambda n: n.startswith("hab_hs"),
    "older seconds-based history/habit": lambda n: n.startswith(("E_", "per_", "per28_", "conv_", "blend")) or n in (
        "reg_weeks", "days_since", "sec14", "sec14_rel", "airings_wk"),
    "context (airtime, category, user level)": lambda n: n in ("exp_rel", "n_air_wk", "airtime_wk", "n_channels", "med_dur", "category",
                                                            "u_sec_wk", "u_top_share", "u_cat_share", "u_hs_wk", "u_hs_top_share"),
}


def covered(names):
    seen = set()
    for f in FAMILIES.values():
        seen |= {n for n in names if f(n)}
    return [n for n in names if n not in seen]


if __name__ == "__main__":
    ctx = at.Ctx(at.FOLDS_ALL)
    names = ctx.names
    print("features not in any family:", covered([names[i] for i in ctx.idx_all]))
    fam = list(FAMILIES)

    def run(label, keep):
        idx = [i for i in ctx.idx_all if keep(names[i])]
        preds = ctx.crossfit("rank_bin", idx)
        sc = {mo: ctx.score(mo, preds[mo]) for mo in at.FOLDS_ALL}
        print(f"{label:70s} features {len(idx):3d}  val12 {sc['val12']:.4f}  mean {np.mean(list(sc.values())):.4f}", flush=True)

    run("ALL features", lambda n: True)
    for k in fam:
        run(f"ALL minus {k}", (lambda kk: (lambda n: not FAMILIES[kk](n)))(k))
    for k in fam:
        run(f"ONLY {k} (+ context)", (lambda kk: (lambda n: FAMILIES[kk](n) or FAMILIES[fam[-1]](n)))(k))
