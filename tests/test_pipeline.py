"""Tests on tiny examples that can be checked by hand. Run:

    python tests/test_pipeline.py

Each test prints what it checks and the numbers, so the output can be compared with a pencil calculation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from build_interactions import overlaps
from common import WATCH_ABS_S, WATCH_FRAC, airings, ap_at_k, candidate_pool, evaluate, pad_to_k
import habits as h

FAILED = []


def check(name, got, expected, tol=1e-4):
    ok = (abs(got - expected) < tol) if isinstance(expected, float) else (got == expected)
    print(f"{'OK  ' if ok else 'FAIL'} {name}: got {got!r}, expected {expected!r}")
    if not ok:
        FAILED.append(name)


# ---- MAP@K: the two examples from the teacher's slides ("MAP@5: Correct recommendations, earlier") ----------
relevant = ["A", "B", "C"]
method_x = ["A", "B", "D", "C", "E"]   # hits at ranks 1, 2, 4 -> precisions 1/1, 2/2, 3/4
method_y = ["D", "A", "E", "B", "C"]   # hits at ranks 2, 4, 5 -> precisions 1/2, 2/4, 3/5
check("slide example, method X: AP@5 = (1.00 + 1.00 + 0.75) / 3", ap_at_k(method_x, relevant), (1 + 1 + 0.75) / 3)
check("slide example, method Y: AP@5 = (0.50 + 0.50 + 0.60) / 3", ap_at_k(method_y, relevant), (0.5 + 0.5 + 0.6) / 3)
check("slide example, both lists: Precision@5 = 3/5", len(set(method_x) & set(relevant)) / 5, 0.6)

# normaliser min(K, |R|): one relevant item found first is a perfect score, found second is 1/2
check("|R| = 1, hit at rank 1 -> AP = 1 / min(5, 1)", ap_at_k(["A", "B", "C", "D", "E"], ["A"]), 1.0)
check("|R| = 1, hit at rank 2 -> AP = (1/2) / min(5, 1)", ap_at_k(["B", "A", "C", "D", "E"], ["A"]), 0.5)
check("|R| = 8 > K, five hits -> AP = 5 / min(5, 8)", ap_at_k([1, 2, 3, 4, 5], list(range(1, 9))), 1.0)
check("no hit -> 0", ap_at_k([9, 8, 7, 6, 5], [1, 2, 3]), 0.0)
check("one hit at rank 4 of |R| = 5 -> (1/4) / 5", ap_at_k([9, 8, 7, 1, 6], [1, 2, 3, 4, 5]), 0.05)
check("empty truth -> 0", ap_at_k([1, 2, 3, 4, 5], []), 0.0)

recs = {1: method_x, 2: method_y, 3: ["Z"] * 5}
truth = {1: relevant, 2: relevant}          # user 3 has no truth -> not evaluated
ev = evaluate(recs, truth)
check("MAP@5 = mean of the two user APs", ev["MAP@5"], ((1 + 1 + 0.75) / 3 + (0.5 + 0.5 + 0.6) / 3) / 2)
check("users without truth are skipped", ev["users"], 2)

# ---- view/schedule overlap ---------------------------------------------------------------------------
ps = np.array([0, 100, 200])
pe = np.array([100, 200, 300])               # programmes A=[0,100) B=[100,200) C=[200,300)
s = np.array([50, 0, 250, 300, 90])
e = np.array([150, 100, 400, 400, 210])      # views
vi, pj, ov = overlaps(ps, pe, s, e)
got = sorted(zip(vi.tolist(), pj.tolist(), ov.tolist()))
expected = [(0, 0, 50), (0, 1, 50), (1, 0, 100), (2, 2, 50), (4, 0, 10), (4, 1, 100), (4, 2, 10)]
check("overlaps: (view, programme, seconds); [0,100) touches B only at the boundary -> no overlap", got, expected)

# ---- causal clipping of windows ------------------------------------------------------------------------
inter = pd.DataFrame({"user_id": np.array([1], dtype=np.uint64), "tv_show_id": [7], "view_start_s": [400],
                      "overlap_s": [600], "prog_start_s": [300], "prog_dur_s": [1000]})   # overlap = [400, 1000)
need = min(WATCH_ABS_S, WATCH_FRAC * 1000)
a = airings(inter, 0, 700)
check(f"window [0,700): only 300 s of the 600 s fall inside (need {need} s) -> watched with 300 s", int(a["sec"].iloc[0]), 300)
check("window [0,650): 250 s < 300 s -> not watched", len(airings(inter, 0, 650)), 0)
a = airings(inter, 700, 2000)
check("window [700,2000): the remaining 300 s count -> watched", int(a["sec"].iloc[0]), 300)

# ---- half-hour slots and time-of-week ----------------------------------------------------------------
idx, slot, sec = h.expand(np.array([0]), np.array([4000]), 0, 10000)
check("interval [0,4000) -> slots with 1800 + 1800 + 400 seconds", sec.tolist(), [1800, 1800, 400])
idx, slot, sec = h.expand(np.array([0]), np.array([4000]), 0, 2000)
check("same interval clipped to [0,2000) -> 1800 + 200", sec.tolist(), [1800, 200])


def how(ts):
    return int(h.how_of(np.datetime64(ts, "s").astype(np.int64) // h.SLOT))


check("Monday 2020-03-09 00:00 -> slot 0 of the week", how("2020-03-09T00:00:00"), 0)
check("Sunday 2020-03-15 23:30 -> slot 335", how("2020-03-15T23:30:00"), 335)
check("Thursday 2020-03-12 19:30 -> 3*48 + 39 = 183", how("2020-03-12T19:30:00"), 183)
check("pooled profile: Monday and Friday 19:30 share class 39",
      (int(h.pooled_of(np.array([39]))[0]), int(h.pooled_of(np.array([4 * 48 + 39]))[0])), (39, 39))
check("pooled profile: Saturday 19:30 -> 48 + 39, Sunday 19:30 -> 96 + 39",
      (int(h.pooled_of(np.array([5 * 48 + 39]))[0]), int(h.pooled_of(np.array([6 * 48 + 39]))[0])), (87, 135))

# ---- lists and pools -----------------------------------------------------------------------------------
check("pad_to_k: [1, 2] + fallback [2, 3, 4, 5, 6] -> five unique ids", pad_to_k([1, 2], [2, 3, 4, 5, 6]), [1, 2, 3, 4, 5])
sched = pd.DataFrame({"tv_show_id": [0, 5, 5, 6, 7], "start_s": [10, 20, 30, 40, 999]})
check("candidate pool: ids 0 and programmes starting outside the window are excluded", candidate_pool(sched, 0, 100).tolist(), [5, 6])

# ---- Kaggle-style truth: committed airings (screen time >= 0.8), exactly five programmes, ties ------------------
import itertools
import hs
from common import screen_airings

# one airing of programme 7 of length 1000 s starting at t=300; the user watches [400, 1000) = 600 s -> screen time 0.6
inter = pd.DataFrame({"user_id": np.array([1, 1], dtype=np.uint64), "channel_id": [3, 3], "tv_show_id": [7, 7],
                      "view_start_s": [400, 5000], "overlap_s": [600, 900], "prog_start_s": [300, 5000], "prog_dur_s": [1000, 1000]})
g = screen_airings(inter, 0, 10_000).sort_values("prog_start_s")
check("screen time = watched / length: 600 / 1000 and 900 / 1000", [round(x, 3) for x in g["screen"]], [0.6, 0.9])
check("window cut at t=5500: the second airing is watched for 500 s -> screen time 0.5", round(float(screen_airings(inter, 0, 5500).sort_values("prog_start_s")["screen"].iloc[-1]), 3), 0.5)

# counts of committed airings per programme for one user: A 5, B 4, C 3, D 2, E 2, F 2 -> the 5th place has count 2
rows = [("A", 5), ("B", 4), ("C", 3), ("D", 2), ("E", 2), ("F", 2)]
cnt = pd.DataFrame({"user_id": np.array([1] * 6, dtype=np.uint64), "tv_show_id": [r[0] for r in rows], "n": [r[1] for r in rows],
                    "first": [10, 20, 30, 40, 50, 60]})
info = hs.tie_info(cnt)[1]
check("tie_info: sure = A, B, C; tied = D, E, F; 2 places left; earliest-first top-5 = A B C D E",
      (info[0], info[1], info[2], info[3]), (["A", "B", "C"], ["D", "E", "F"], 2, ["A", "B", "C", "D", "E"]))
check("a user with only 4 committed programmes has no truth (strictly five)", len(hs.tie_info(cnt[cnt["tv_show_id"].isin(list("ABCD"))])), 0)
# expected AP of the list [A, B, C, D, X] over the three equally likely truths {A,B,C,D,E}, {A,B,C,D,F}, {A,B,C,E,F}
rec = ["A", "B", "C", "D", "X"]
exact = np.mean([ap_at_k(rec, set("ABC") | set(c)) for c in itertools.combinations("DEF", 2)])
samples = hs.truth_samples({1: info}, draws=3000, seed=0)
got = hs.evaluate_samples({1: rec}, samples, [1])["MAP@5"]
check("expected AP of [A,B,C,D,X] over the 3 equally likely truths = (0.8 + 0.8 + 0.6) / 3 = 0.7333 "
      "(truths containing D: four hits at ranks 1-4 -> 4/5; the truth without D: three hits -> 3/5)",
      round(got, 2), round(float(exact), 2))
check("... and the hand value itself", round(float(exact), 4), round((0.8 + 0.8 + 0.6) / 3, 4))

print()
print("ALL TESTS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
