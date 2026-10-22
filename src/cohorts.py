"""Who is in the evaluation cohort and who is not: a disjoint decomposition of the 1,260 template users for every fold.

Two properties of a user in a fold are crossed:
  history : the user has at least one committed airing (screen time >= 0.8) before the cut-off (otherwise no personalisation is
            possible and the popularity fallback is used)
  truth   : the user has at least five committed programmes in the target window (the host keeps exactly five per user, so users
            with fewer are not in the solution file and are excluded from the replica)
The four cells are disjoint and add up to 1,260. For the test window every user has a truth by construction (the host's file).
Output: results/cohorts.md
"""
import numpy as np
import pandas as pd

import hs
from common import FOLDS, ROOT, Split, screen_airings, test_users

if __name__ == "__main__":
    tmpl = set(test_users().tolist())
    lines = ["# Cohorts of the 1,260 template users (disjoint decomposition)\n",
             "| fold | history window | target window | with history and truth (evaluated) | with history, no truth | no history, with truth | no history, no truth | total |",
             "|---|---|---|---|---|---|---|---|"]
    for mode in list(FOLDS):
        sp = Split(mode)
        g = screen_airings(sp.inter, sp.train_lo, sp.train_hi)
        hist = set(g[g["screen"] >= 0.8]["user_id"].unique().tolist()) & tmpl
        if mode == "test":
            truth = set(tmpl)
        else:
            truth = set(hs.tie_info(hs.qualifying_counts(sp.inter, sp.tgt_lo, sp.tgt_hi)).keys()) & tmpl
        a = len(hist & truth)
        b = len(hist - truth)
        c = len(truth - hist)
        d = len(tmpl - hist - truth)
        assert a + b + c + d == len(tmpl) == 1260
        f = lambda s: pd.to_datetime(s, unit="s").strftime("%Y-%m-%d")
        lines.append(f"| {mode} | {f(sp.train_lo)} .. {f(sp.train_hi)} | {f(sp.tgt_lo)} .. {f(sp.tgt_hi)} | {a} | {b} | {c} | {d} | {a + b + c + d} |")
    lines.append("\nFor `test` the 'truth' column is true by construction (every user of the solution file has exactly five programmes), so the cell "
                 "'with history, no truth' is empty there; users without a committed history get the popularity fallback.")
    text = "\n".join(lines)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "cohorts.md").write_text(text, encoding="utf-8")
    print(text)
