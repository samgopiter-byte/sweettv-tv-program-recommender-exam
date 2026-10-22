"""Diagnostic: how much of the validation truth is simply the user's own habitual programmes?"""
import numpy as np
from common import Split, evaluate, user_item_decay, K, pad_to_k
import methods as m

sp = Split("val")
fb = m.pop_ranking(sp.train_ui, sp.pool, "seconds")[:20]
pool = set(sp.pool.tolist())
users = sp.users

def personal(ui):
    ui = ui[ui["tv_show_id"].isin(pool)].sort_values(["user_id", "sec"], ascending=[True, False])
    top = {u: g["tv_show_id"].tolist()[:K] for u, g in ui.groupby("user_id", sort=False)}
    return {u: pad_to_k(top.get(u, []), fb) for u in users}

print("personal top by all-time seconds:", evaluate(personal(sp.train_ui), sp.truth, users))
for h in (7, 14, 28, 56):
    ui = user_item_decay(sp.inter, sp.train_lo, sp.train_hi, h)
    print(f"personal top, decay half-life {h:2d} d:", evaluate(personal(ui), sp.truth, users))

# how much of truth is already in the user's training history / in the pool / unseen in training
hist = sp.train_ui.groupby("user_id")["tv_show_id"].apply(set).to_dict()
tot = rep = poolhit = 0
for u in users:
    for it in sp.truth.get(u, []):
        tot += 1
        rep += it in hist.get(u, set())
        poolhit += it in pool
print("truth items", tot, "| already watched in training", rep, f"({100*rep/tot:.1f}%)", "| in candidate pool", poolhit, f"({100*poolhit/tot:.1f}%)")
