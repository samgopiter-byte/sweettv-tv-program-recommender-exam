"""Which re-ranker variant keeps its gain over the plain blend when the window changes?

For every variant: gain over the blend (same users) for (a) in-fold 5-fold cross-fit by users and
(b) train on the other validation fold -> evaluate on this one (user-disjoint, time-shifted).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import evaluate
import rerank as rr

CORE = ["blend", "blend_rank", "E_f56_rel", "E_f56_rank", "E_f28_rel", "E_p56_rel", "per_rel", "per_rank", "per28_rel",
        "conv_rel", "conv_rank", "exp_rel", "reg_weeks", "days_since", "sec14_rel", "airings_wk", "u_cat_share", "pop_rel", "als_rel"]
HIST_DEP = ["mom14", "mom28", "log_user_frac", "span_frac"]


class Gbdt:
    def __init__(self, leaves, rounds, min_data=100):
        self.p, self.r = dict(num_leaves=leaves, min_data_in_leaf=min_data), rounds

    def fit(self, X, y, rows):
        self.m = rr.fit(X, y, rows, self.r, self.p)
        return self

    def score(self, X):
        return self.m.predict(X)


class Logit:
    def __init__(self, C=0.3):
        self.C = C

    def fit(self, X, y, rows):
        self.sc = StandardScaler().fit(X)
        self.m = LogisticRegression(C=self.C, max_iter=1000).fit(self.sc.transform(X), y)
        return self

    def score(self, X):
        return self.m.decision_function(self.sc.transform(X))


def run_variant(name, make, feats, data, fo):
    out = {}
    for src, dst in (("val2", "val"), ("val", "val2")):
        sp, M, fb, Dd = data[dst]
        Ds = data[src][3]
        names = list(Dd["names"])
        idx = [names.index(f) for f in feats] if feats else list(range(len(names)))
        users_te = [u for i, u in enumerate(sp.users) if fo[i] >= 3]
        bl = Dd["X"][:, names.index("blend")]
        base = evaluate(rr.lists_from_scores(sp, M, fb, Dd["rows"], Dd["cols"], bl), sp.truth, users_te)["MAP@5"]
        tr = fo[Ds["rows"]] < 3
        mod = make().fit(Ds["X"][tr][:, idx], Ds["y"][tr], Ds["rows"][tr])
        te = fo[Dd["rows"]] >= 3
        sc = mod.score(Dd["X"][te][:, idx])
        shifted = evaluate(rr.lists_from_scores(sp, M, fb, Dd["rows"][te], Dd["cols"][te], sc), sp.truth, users_te)["MAP@5"]
        pred = np.zeros(len(Dd["rows"]))
        for f in range(5):
            trn = fo[Dd["rows"]] != f
            mm = make().fit(Dd["X"][trn][:, idx], Dd["y"][trn], Dd["rows"][trn])
            pred[~trn] = mm.score(Dd["X"][~trn][:, idx])
        infold = evaluate(rr.lists_from_scores(sp, M, fb, Dd["rows"], Dd["cols"], pred), sp.truth, users_te)["MAP@5"]
        out[dst] = (base, shifted, infold)
    g_sh = np.mean([v[1] - v[0] for v in out.values()])
    g_in = np.mean([v[2] - v[0] for v in out.values()])
    print(f"{name:34s} gain over blend: shifted {g_sh:+.4f} | in-fold {g_in:+.4f}   "
          f"(val: blend {out['val'][0]:.4f} shifted {out['val'][1]:.4f} infold {out['val'][2]:.4f}; "
          f"val2: blend {out['val2'][0]:.4f} shifted {out['val2'][1]:.4f} infold {out['val2'][2]:.4f})", flush=True)


if __name__ == "__main__":
    data = {mode: rr.get_D(mode) for mode in ("val", "val2")}
    n_users = len(data["val"][1].sel)
    fo = rr.user_folds(n_users)
    allnames = list(data["val"][3]["names"])
    no_hist = [n for n in allnames if n not in HIST_DEP]
    run_variant("gbdt L15 R150 all", lambda: Gbdt(15, 150), None, data, fo)
    run_variant("gbdt L4 R100 all", lambda: Gbdt(4, 100), None, data, fo)
    run_variant("gbdt L4 R60 all", lambda: Gbdt(4, 60), None, data, fo)
    run_variant("gbdt L7 R100 no hist-dependent", lambda: Gbdt(7, 100), no_hist, data, fo)
    run_variant("gbdt L4 R100 no hist-dependent", lambda: Gbdt(4, 100), no_hist, data, fo)
    run_variant("gbdt L7 R100 core", lambda: Gbdt(7, 100), CORE, data, fo)
    run_variant("gbdt L4 R100 core", lambda: Gbdt(4, 100), CORE, data, fo)
    run_variant("logit C0.3 core", lambda: Logit(0.3), CORE, data, fo)
    run_variant("logit C0.3 no hist-dependent", lambda: Logit(0.3), no_hist, data, fo)
