"""Popularity baseline on the validation split, plus split statistics used in the report."""
import numpy as np
import pandas as pd
from common import Split, evaluate, K


def popularity_ranking(train_ui, pool, by):
    ui = train_ui[train_ui["tv_show_id"].isin(pool)]
    if by == "users":
        sc = ui.groupby("tv_show_id")["user_id"].nunique()
    else:
        sc = ui.groupby("tv_show_id")["sec"].sum()
    return sc.sort_values(ascending=False).index.tolist()


if __name__ == "__main__":
    sp = Split("val")
    users = sp.users
    truth_users = [u for u in users if u in sp.truth]
    print("validation: train views before", pd.to_datetime(sp.train_hi, unit="s"), "| target window starts", pd.to_datetime(sp.tgt_lo, unit="s"))
    print("test users", len(users), "| with a validation truth", len(truth_users),
          "| with train history", len([u for u in users if u in sp.train_users]),
          "| truth users with train history", len([u for u in truth_users if u in sp.train_users]))
    print("candidate pool (labelled programmes airing in the target window):", len(sp.pool))
    print("train user-item rows", len(sp.train_ui), "| truth size mean", np.mean([len(v) for v in sp.truth.values()]))

    for by in ["users", "seconds"]:
        rank = popularity_ranking(sp.train_ui, sp.pool, by)
        rec = {u: rank[:K] for u in users}
        print(f"popularity by {by}:", evaluate(rec, sp.truth, users), "top5", rank[:K])
