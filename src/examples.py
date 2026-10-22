"""Two example users for the business analysis of the report (fold val12: 8-week history, 12-week target).

Users are chosen by a rule, not by hand: among the template users that have a truth in the target window and some history,
the 'heavy' viewer is the one closest to the 90th percentile of committed airings in the history, the 'light' viewer the one
closest to the 15th percentile. For each user: history (most committed programmes), the true top 5 of the target window with
counts, and the top 5 of every method with id, title, category and a mark for hits. Output: examples_two_users.md.
"""
import numpy as np
import pandas as pd

import attempts2 as at
import hs
import methods as m
import rerank as rr
from common import ROOT, Split, K, ap_at_k

MODE = "val12"


def main():
    ctx = at.Ctx(at.FOLDS_ALL)
    sp, M, fb, D = ctx.data[MODE]
    info = ctx.ev[MODE]["info"]
    samples = ctx.ev[MODE]["samples"]
    users = ctx.ev[MODE]["users"]
    allov = pd.read_parquet(rr.PROC + "all_overlaps.parquet")
    X = hs.committed_matrix_all(sp, M)
    Mc = m.with_input(M, X)
    hist_cnt = np.asarray(X[M.sel].sum(axis=1)).ravel()
    pos = {u: i for i, u in enumerate(sp.users)}
    cand = [u for u in users if hist_cnt[pos[u]] > 0]
    vals = np.array([hist_cnt[pos[u]] for u in cand])
    heavy = cand[int(np.argmin(np.abs(vals - np.percentile(vals, 90))))]
    light = cand[int(np.argmin(np.abs(vals - np.percentile(vals, 15))))]

    S = {}
    S["Popularity"] = m.pop_scores(M)
    S["MF (ALS, committed input)"] = m.als_scores(Mc, factors=32, reg=50.0, alpha=5.0)
    S["AR (association rules)"] = m.assoc_scores(Mc, min_supp=10, beta=0.3)
    S["CBF (tf-idf attributes)"] = m.cbf_scores(Mc)
    S["PPR (co-view graph)"] = m.ppr_scores(M, X, d=0.85)
    S["Slot habit (committed)"] = hs.habit_hs(sp, M, allov, half_life=56, pooled=True)
    course = [S[k] for k in ("MF (ALS, committed input)", "AR (association rules)", "CBF (tf-idf attributes)", "PPR (co-view graph)")]
    mats = sum(1.0 / (60.0 + rr.rank_matrix(s, M.pool_ix)) for s in course)
    S["Rank fusion (ALS+AR+CBF+PPR)"] = mats
    lists = {k: m.recommend(M, v, fb) for k, v in S.items()}
    # final re-ranker: cross-fit ensemble (rank_bin + rank_soft) on the same fold
    preds = [ctx.crossfit(kind, ctx.idx_all) for kind in ("rank_bin", "rank_soft")]
    z = sum(at.zscore(p[MODE], D["rows"]) for p in preds)
    lists["Final hybrid (re-ranker)"] = rr.lists_from_scores(sp, M, fb, D["rows"], D["cols"], z)

    sched = sp.sched[sp.sched["tv_show_id"] != 0].drop_duplicates("tv_show_id").set_index("tv_show_id")
    title = lambda i: str(sched.loc[i, "tv_show_title"])[:44] if i in sched.index else "?"
    cat = lambda i: str(sched.loc[i, "tv_show_category"]) if i in sched.index else "?"
    cnt_hist = hs.qualifying_counts(sp.inter, sp.train_lo, sp.train_hi).set_index(["user_id", "tv_show_id"])["n"]
    cnt_tgt = hs.qualifying_counts(sp.inter, sp.tgt_lo, sp.tgt_hi).set_index(["user_id", "tv_show_id"])["n"]

    out = [f"# Two example users (fold {MODE}: history {pd.to_datetime(sp.train_lo, unit='s').date()} .. {pd.to_datetime(sp.train_hi, unit='s').date()}, "
           f"target {pd.to_datetime(sp.tgt_lo, unit='s').date()} .. {pd.to_datetime(sp.tgt_hi, unit='s').date()})\n",
           f"Selection rule: among {len(cand)} template users that have a truth and some history, the heavy viewer is closest to the 90th percentile of "
           f"committed airings in the history ({np.percentile(vals, 90):.0f}), the light viewer closest to the 15th percentile ({np.percentile(vals, 15):.0f}).\n"]
    for label, u in (("Heavy viewer", heavy), ("Light viewer", light)):
        sure, tied, need, det = info[u]
        truth_ids = set(sure) | set(tied)
        out.append(f"\n## {label}: user {u} (committed airings in the history: {hist_cnt[pos[u]]:.0f})\n")
        h = cnt_hist[u].sort_values(ascending=False).head(8)
        out.append("History, most committed programmes (count of airings watched >= 80%): " +
                   "; ".join(f"{i} {title(i)} [{cat(i)}] x{int(n)}" for i, n in h.items()) + "\n")
        t = cnt_tgt[u].sort_values(ascending=False).head(8)
        out.append("Truth in the target window (top by count; the 5th place may be tied): " +
                   "; ".join(f"{i} {title(i)} [{cat(i)}] x{int(n)}" for i, n in t.items()) + "\n")
        out.append(f"Truth top-5 used for scoring: sure = {sure}, tied at the cut = {tied} (places left {need}).\n")
        out.append("| Method | Top-5 (id, title, category; * = in the truth) | expected AP@5 |\n|---|---|---|")
        for name, L in lists.items():
            rec = L[u]
            ap = np.mean([ap_at_k(rec, s) for s in samples[u]])
            cells = "<br>".join(f"{k + 1}. {i}{'*' if i in truth_ids else ''} {title(i)} [{cat(i)}]" for k, i in enumerate(rec))
            out.append(f"| {name} | {cells} | {ap:.3f} |")
    (ROOT / "examples_two_users.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out)[:6000])


if __name__ == "__main__":
    main()
