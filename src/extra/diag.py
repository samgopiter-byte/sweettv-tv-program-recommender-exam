"""Where does the final model lose? Error analysis on the replica (cross-fit LightGBM rank_bin predictions)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import lite
import div
from common import ap_at_k

if __name__ == "__main__":
    L = lite.Lite("v6")
    preds = div.crossfit(L, "lgb_bin")
    rows_out = []
    for mo in lite.FOLDS:
        me, D = L.meta[mo], L.D[mo]
        lists = L.lists(mo, preds[mo])
        col_of = {int(it): j for j, it in enumerate(me["items"])}
        pos = {u: i for i, u in enumerate(me["users"])}
        cand = pd.Series(D["cols"]).groupby(D["rows"]).apply(set)
        t_seen = t_new = hit_seen = hit_new = cand_seen = cand_new = 0
        ap_user, heavy, tied_flag = [], [], []
        for u in me["eligible"]:
            i = pos[u]
            sure, tied, need, det = me["info"][u]
            truth_all = set(sure) | set(tied)
            for it in truth_all:
                seen = col_of.get(it, -1) in me["hist_items"][i]
                inl = col_of.get(it, -1) in cand.get(i, set())
                got = it in lists[u]
                if seen:
                    t_seen += 1; cand_seen += inl; hit_seen += got
                else:
                    t_new += 1; cand_new += inl; hit_new += got
            ap_user.append(np.mean([ap_at_k(lists[u], s) for s in me["samples"][u]]))
            heavy.append(me["hist_cnt"][i])
            tied_flag.append(len(tied) > need)
        ap_user = np.array(ap_user); heavy = np.array(heavy); tied_flag = np.array(tied_flag)
        q = np.quantile(heavy, [0.25, 0.5, 0.75])
        grp = np.digitize(heavy, q)
        rows_out.append(dict(fold=mo, users=len(ap_user), MAP=ap_user.mean(),
                             share_seen=t_seen / (t_seen + t_new), cand_seen=cand_seen / max(t_seen, 1), cand_new=cand_new / max(t_new, 1),
                             hit_seen=hit_seen / max(t_seen, 1), hit_new=hit_new / max(t_new, 1),
                             MAP_ties=ap_user[tied_flag].mean(), MAP_noties=ap_user[~tied_flag].mean(), share_ties=tied_flag.mean(),
                             **{f"MAP_q{k + 1}": ap_user[grp == k].mean() for k in range(4)}))
    df = pd.DataFrame(rows_out).set_index("fold")
    pd.set_option("display.width", 250)
    print(df.round(3).T.to_string())
    print("\nmean over folds:\n", df.mean().round(3).to_string())
