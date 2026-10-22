"""Final strategy v2 and the two candidate submissions.

* blend      : minmax(habit model E_f56) + minmax(personal history x exposure)             (2 parameters)
* ltr (final): LightGBM LambdaRank re-ranker over candidates from the habit, personal-history, conversion, ALS,
               association-rule, popularity and CBF scorers. Trained on the labels of both validation folds
               (users of fold val: history < 2020-06-01, target 06-01..07-26; fold val2: history < 05-04,
               target 05-04..06-28), applied to the test setting (history < 2020-07-27).
Features that grow with the length of the history (momentum ratios, user share, span) are excluded because the
test history (20 weeks) is longer than both training histories (8 and 12 weeks); see rerank_shift.py.
"""
import numpy as np
import pandas as pd

from common import ROOT, K, evaluate
import rerank as rr

HIST_DEP = ["mom14", "mom28", "log_user_frac", "span_frac"]
ROUNDS = 100
LGB = dict(num_leaves=7, min_data_in_leaf=100)
SUB = ROOT / "submissions"


def feature_idx(names):
    return [i for i, n in enumerate(names) if n not in HIST_DEP]


def train_final(Ds):
    names = list(Ds[0]["names"])
    idx = feature_idx(names)
    n_users = int(max(D["rows"].max() for D in Ds)) + 1
    X = np.vstack([D["X"][:, idx] for D in Ds])
    y = np.concatenate([D["y"] for D in Ds])
    rows = np.concatenate([D["rows"] + k * n_users for k, D in enumerate(Ds)])
    return rr.fit(X, y, rows, ROUNDS, LGB), idx, names


def write_submission(sp, lists, path):
    tmpl = pd.read_csv(ROOT / "data" / "raw" / "submission.csv", dtype={"user_id": "uint64"})
    tmpl["tv_show_id"] = [" ".join(str(x) for x in lists[u]) for u in tmpl["user_id"]]
    tmpl.to_csv(path, index=False)
    return tmpl


if __name__ == "__main__":
    SUB.mkdir(exist_ok=True)
    data = {mode: rr.get_D(mode) for mode in ("val", "val2")}
    Ds = [data["val"][3], data["val2"][3]]
    n_users = len(data["val"][1].sel)

    # honest validation numbers: 5-fold cross-fit by users, trained on the other users of both folds
    names = list(Ds[0]["names"])
    idx = feature_idx(names)
    Ds_sub = [dict(D, X=D["X"][:, idx]) for D in Ds]
    preds, _ = rr.crossfit_pooled(Ds_sub, n_users, rounds=ROUNDS, params=LGB)
    for mode, D, p in zip(("val", "val2"), Ds, preds):
        sp, M, fb, _ = data[mode]
        bl = D["X"][:, names.index("blend")]
        r_bl = evaluate(rr.lists_from_scores(sp, M, fb, D["rows"], D["cols"], bl), sp.truth, sp.users)
        r_lt = evaluate(rr.lists_from_scores(sp, M, fb, D["rows"], D["cols"], p), sp.truth, sp.users)
        print(f"{mode}: blend MAP@5 {r_bl['MAP@5']:.4f} Hit@5 {r_bl['HitRate@5']:.3f} | LTR cross-fit MAP@5 {r_lt['MAP@5']:.4f} Hit@5 {r_lt['HitRate@5']:.3f}")

    model, idx, names = train_final(Ds)
    imp = sorted(zip(model.feature_importance("gain"), [names[i] for i in idx]), reverse=True)
    print("top features by gain:", [(n, round(float(g) / sum(x for x, _ in imp), 3)) for g, n in imp[:10]])

    sp, M, fb, D = rr.get_D("test")
    pred = model.predict(D["X"][:, idx])
    lists_ltr = rr.lists_from_scores(sp, M, fb, D["rows"], D["cols"], pred)
    bl = D["X"][:, list(D["names"]).index("blend")]
    lists_bl = rr.lists_from_scores(sp, M, fb, D["rows"], D["cols"], bl)
    t1 = write_submission(sp, lists_ltr, SUB / "submission_ltr_v2.csv")
    write_submission(sp, lists_bl, SUB / "submission_blend_v2.csv")
    v1 = pd.read_csv(SUB / "submitted_v1_ref57530793.csv", dtype={"user_id": "uint64", "tv_show_id": str})
    v1l = {u: [int(x) for x in s.split()] for u, s in zip(v1["user_id"], v1["tv_show_id"])}
    same = lambda A, B: np.mean([set(A[u]) == set(B[u]) for u in sp.users])
    ov = lambda A, B: np.mean([len(set(A[u]) & set(B[u])) / K for u in sp.users])
    print(f"test: users {len(t1)}, candidate rows {len(D['rows'])}")
    print(f"  share of users with the same top-5 SET: ltr vs blend {same(lists_ltr, lists_bl):.3f}, ltr vs v1 {same(lists_ltr, v1l):.3f}, blend vs v1 {same(lists_bl, v1l):.3f}")
    print(f"  mean overlap of top-5: ltr vs blend {ov(lists_ltr, lists_bl):.3f}, ltr vs v1 {ov(lists_ltr, v1l):.3f}")
    pool = set(sp.pool.tolist())
    for name, L in (("ltr", lists_ltr), ("blend", lists_bl)):
        bad = [u for u in sp.users if len(set(L[u])) != K or not set(L[u]) <= pool]
        print(f"  {name}: lists that are not 5 unique pool ids: {len(bad)}")
