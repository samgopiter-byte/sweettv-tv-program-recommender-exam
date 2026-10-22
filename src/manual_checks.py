"""Numbers for the student's own manual checks (report section "What I personally checked").

Nothing here claims that anything was verified. For every check the script prints the raw data needed to recompute a
number with a calculator, a blank to fill in, and - after a separator - the answer of the code. The student does the
arithmetic first and compares afterwards. Output: MANUAL_CHECKS.md.

    python manual_checks.py screen truth habit ap
"""
import sys
import numpy as np
import pandas as pd

import hs
from common import ROOT, W11, W23, Split, ap_at_k, load_interactions, load_schedule, screen_airings, test_users

RAWD = ROOT / "data" / "raw"
SEP = "\n> ---- the code's answer (look only after you have computed your own) ----\n"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
out = []


def t(s):
    return pd.to_datetime(s, unit="s").strftime("%d.%m.%Y %H:%M:%S")


def say(s=""):
    out.append(s)
    print(s, flush=True)


# ---------------------------------------------------------------------------------------------------------------
def check_screen():
    """One airing watched in two pieces: screen time = watched seconds / airing length (committed if >= 0.8)."""
    inter = load_interactions()
    g = screen_airings(inter, W11, W23)
    nv = inter.groupby(["user_id", "channel_id", "tv_show_id", "prog_start_s"]).size().rename("nviews").reset_index()
    g = g.merge(nv, on=["user_id", "channel_id", "tv_show_id", "prog_start_s"])
    g = g[(g["nviews"] == 2) & (g["dur"].between(1500, 3600)) & g["user_id"].isin(test_users())]
    sch_all = pd.concat([pd.read_csv(RAWD / f, encoding="utf-8") for f in ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv")])
    views_all = pd.read_csv(RAWD / "dataset11-30.csv", dtype={"user_id": "uint64"})
    say("## Check 1. Screen time of two airings (recompute by hand)\n")
    say("Rule of the host (Kaggle Data page): screen time = time the user watched the airing / length of the airing; "
        "only airings with screen time >= 0.8 count as 'committed'. Two examples, one clearly above and one clearly below the threshold.\n")
    for label, target in (("A", 0.86), ("B", 0.74)):
        row = g.assign(gap=(g["screen"] - target).abs()).sort_values(["gap", "user_id", "prog_start_s"]).iloc[0:1]
        # read the values column by column: a row Series would turn the 64-bit user id into a float and corrupt it
        u, ch, pid = int(row["user_id"].iloc[0]), int(row["channel_id"].iloc[0]), int(row["tv_show_id"].iloc[0])
        start, dur, sec, screen = int(row["prog_start_s"].iloc[0]), int(row["dur"].iloc[0]), int(row["sec"].iloc[0]), float(row["screen"].iloc[0])
        sch = sch_all[(sch_all["channel_id"] == ch) & (sch_all["start_time"] == t(start)) & (sch_all["tv_show_id"] == pid)].iloc[0]
        a0 = pd.to_datetime(sch["start_time"], format="%d.%m.%Y %H:%M:%S")
        views = views_all[(views_all["user_id"] == np.uint64(u)) & (views_all["vsetv_id"] == ch)]
        views = views[(pd.to_datetime(views["stop_time"]) > a0) & (pd.to_datetime(views["start_time"]) < a0 + pd.Timedelta(seconds=dur))]
        say(f"### Example {label}\n")
        say(f"**Airing** (from the raw schedule file): channel {ch} \"{sch['channel_title']}\", programme {pid} \"{sch['tv_show_title']}\", "
            f"starts **{sch['start_time']}**, length **{dur} s** (so it ends {t(start + dur)}).\n")
        say(f"**User** {u}. Views of this user on this channel that overlap the airing (raw rows of `dataset11-30.csv`):\n")
        say("| # | start_time | stop_time | overlap with the airing, seconds (fill in) |\n|---|---|---|---|")
        for i, v in enumerate(views.itertuples(), 1):
            say(f"| {i} | {v.start_time} | {v.stop_time} | ____ |")
        say("\nYour calculation: overlap = min(stop, airing end) - max(start, airing start) for each view; "
            f"sum = ____ s; screen time = sum / {dur} = ____; committed (>= 0.8)? ____")
        say(SEP + f"seconds {sec}, length {dur}, screen time {screen:.4f}, committed: {screen >= 0.8}\n")


# ---------------------------------------------------------------------------------------------------------------
def pick_truth_user(sp):
    """Among the template users whose truth has a 2-way tie for the last place, the one with the fewest committed airings
    in the window (at least 12), so that the counting by hand stays short."""
    cnt = hs.qualifying_counts(sp.inter, sp.tgt_lo, sp.tgt_hi)
    info = hs.tie_info(cnt)
    total = cnt.groupby("user_id")["n"].sum()
    tmpl = set(test_users().tolist())
    best = None
    for u in sorted(info):
        sure, tied, need, det = info[u]
        if u in tmpl and len(tied) == 2 and need == 1 and len(sure) == 4 and total[u] >= 12:
            if best is None or total[u] < total[best]:
                best = u
    if best is None:
        raise RuntimeError("no user with a 2-way tie for one place")
    return best, info[best]


def check_truth(sp):
    """The hidden truth of one user: count committed airings per programme, keep the five most frequent."""
    u, (sure, tied, need, det) = pick_truth_user(sp)
    g = screen_airings(sp.inter, sp.tgt_lo, sp.tgt_hi)
    g = g[(g["user_id"] == u) & (g["screen"] >= 0.8)]
    sched = sp.sched[sp.sched["tv_show_id"] != 0].drop_duplicates("tv_show_id").set_index("tv_show_id")
    say(f"## Check 2. The truth of one user: top five programmes by the number of committed airings (user {u}, "
        f"window {t(sp.tgt_lo)[:10]} .. {t(sp.tgt_hi)[:10]})\n")
    say("Below are ALL airings of this user in the window with screen time >= 0.8 (programme, start of the airing, screen time). "
        "Count the airings per programme, order the programmes by count, and take the top five. Ties at the 5th place are possible.\n")
    say("| programme id | title | airing start | screen time |\n|---|---|---|---|")
    for r in g.sort_values(["tv_show_id", "prog_start_s"]).itertuples():
        say(f"| {r.tv_show_id} | {str(sched.loc[r.tv_show_id, 'tv_show_title'])[:40]} | {t(r.prog_start_s)} | {r.screen:.2f} |")
    say("\nYour counts per programme: ____ ; the five most frequent: ____ ; is there a tie at the 5th place? ____")
    cnt = g.groupby("tv_show_id").size().sort_values(ascending=False)
    say(SEP + "counts per programme: " + ", ".join(f"{i}: {n}" for i, n in cnt.items()))
    say(f"strictly above the cut (always in the truth): {sure}; tied at the cut: {tied}; places left for the tied: {need}\n")
    return u


# ---------------------------------------------------------------------------------------------------------------
def check_habit(sp, u):
    """Expected number of committed airings of one programme from the slot habit: sum over its airings of hits / (airings + 1)."""
    import habits as h
    allov = pd.read_parquet(ROOT / "data" / "processed" / "all_overlaps.parquet")
    import methods as m
    M = m.Matrices(sp)
    pos = {x: i for i, x in enumerate(sp.users)}
    info = hs.tie_info(hs.qualifying_counts(sp.inter, sp.tgt_lo, sp.tgt_hi))[u]
    cand = sorted(set(info[0]) | set(info[1]))
    sched = sp.sched
    hist = sched[(sched["start_s"] >= sp.train_lo) & (sched["start_s"] < sp.train_hi)]
    tgt = sched[(sched["tv_show_id"] != 0) & (sched["start_s"] >= sp.tgt_lo) & (sched["start_s"] < sp.tgt_hi)]

    def cell(ch, s):
        slot = (s + 900) // 1800
        how = int(h.how_of(np.array([slot]))[0])
        return ch, int(h.pooled_of(np.array([how]))[0])

    def label(c):
        k = c[1]
        cls = ["Mon-Fri", "Sat", "Sun"][k // 48]
        return f"{cls} {(k % 48) // 2:02d}:{30 * (k % 2):02d}"

    a = screen_airings(allov, sp.train_lo, sp.train_hi)
    a = a[(a["user_id"] == u) & (a["screen"] >= 0.8)]
    R = {}
    for r in a.itertuples():
        c = cell(r.channel_id, r.prog_start_s)
        R[c] = R.get(c, 0) + 1
    Z = {}
    for r in hist.itertuples():
        c = cell(r.channel_id, r.start_s)
        Z[c] = Z.get(c, 0) + 1
    # a programme with few airings in the window (so that the sum can be done by hand) for which the user's rate is positive;
    # among those, the one with the largest expected count (ties: the smaller id)
    counts = tgt.groupby("tv_show_id").size()
    few = counts[(counts >= 4) & (counts <= 8)].index
    best, best_val = None, 0.0
    for pid, grp in tgt[tgt["tv_show_id"].isin(few)].groupby("tv_show_id"):
        val = sum(R.get(cell(r.channel_id, r.start_s), 0) / (Z.get(cell(r.channel_id, r.start_s), 0) + 1.0) for r in grp.itertuples())
        if val > best_val + 1e-12:
            best, best_val = pid, val
    rows = tgt[tgt["tv_show_id"] == best].sort_values("start_s")
    title = str(rows["tv_show_title"].iloc[0])[:44]
    say(f"## Check 3. Expected committed airings of one programme from the slot habit (user {u}, programme {best} \"{title}\")\n")
    say("Model: for each airing of the programme in the target window take the user's rate in the airing's channel and time slot "
        "(slot = weekday class Mon-Fri / Sat / Sun and the half hour of the start, rounded to the nearest half hour): "
        "rate = hits / (airings + 1), where *hits* = airings in that channel and slot in the history window that the user watched "
        "for >= 80% and *airings* = all airings starting in that channel and slot in the history window. The expected number of "
        "committed airings of the programme is the sum of the rates over its airings.\n")
    say("| airing start | channel | slot | hits (history) | airings (history) | rate = hits / (airings + 1) (fill in) |\n|---|---|---|---|---|---|")
    total = 0.0
    for r in rows.itertuples():
        c = cell(r.channel_id, r.start_s)
        total += R.get(c, 0) / (Z.get(c, 0) + 1.0)
        say(f"| {t(r.start_s)} ({DAYS[(pd.to_datetime(r.start_s, unit='s').dayofweek)]}) | {r.channel_id} | {label(c)} | {R.get(c, 0)} | {Z.get(c, 0)} | ____ |")
    say("\nYour sum of the rates = ____ (expected number of committed airings of this programme for this user)")
    full = hs.habit_hs(sp, M, allov, pooled=True)
    say(SEP + f"sum of rates computed here: {total:.4f}; value of the model's scorer `hab_hs_p` for this user and programme: "
        f"{full[pos[u], M.item_ix[best]]:.4f}\n")


# ---------------------------------------------------------------------------------------------------------------
def check_ap(sp):
    """AP@5 by hand for two users of the fold val12 with the final ensemble (one without ties, one with a 2-way tie)."""
    import attempts2 as at
    import rerank as rr
    ctx = at.Ctx(at.FOLDS_ALL)
    spv, M, fb, D = ctx.data["val12"]
    info = ctx.ev["val12"]["info"]
    preds = [ctx.crossfit(kind, ctx.idx_all) for kind in ("rank_bin", "rank_soft")]
    z = sum(at.zscore(p["val12"], D["rows"]) for p in preds)
    lists = rr.lists_from_scores(spv, M, fb, D["rows"], D["cols"], z)
    sched = spv.sched[spv.sched["tv_show_id"] != 0].drop_duplicates("tv_show_id").set_index("tv_show_id")
    ttl = lambda i: f"{i} {str(sched.loc[i, 'tv_show_title'])[:36]}" if i in sched.index else str(i)
    tmpl = set(test_users().tolist())
    chosen = {}
    for u in sorted(info):
        if u not in tmpl:
            continue
        sure, tied, need, det = info[u]
        hits = len(set(lists[u]) & (set(sure) | set(tied)))
        if "plain" not in chosen and need == len(tied) and hits >= 3:
            chosen["plain"] = u
        # the list must contain exactly one of the two tied programmes, otherwise both possible truths give the same AP
        if "tie" not in chosen and need == 1 and len(tied) == 2 and len(set(lists[u]) & set(sure)) >= 2 and len(set(lists[u]) & set(tied)) == 1:
            chosen["tie"] = u
        if len(chosen) == 2:
            break
    say("## Check 4. AP@5 by hand (final ensemble, fold val12)\n")
    say("AP@5 = (sum of the precisions at the ranks of the hits) / 5, because every user has exactly five programmes in the truth "
        "(the teacher's slides: normaliser min(K, |R|) = 5). Example from the slides: hits at ranks 1, 2, 4 -> (1/1 + 2/2 + 3/4) / 3 for a user with 3 relevant items.\n")
    for kind, u in chosen.items():
        sure, tied, need, det = info[u]
        say(f"### User {u} ({'no randomness in the truth' if kind == 'plain' else 'a tie decides one place of the truth'})\n")
        say("Recommended list (rank, programme):\n")
        for k, i in enumerate(lists[u], 1):
            say(f"{k}. {ttl(i)}")
        if kind == "plain":
            truth = set(sure) | set(tied)
            say("\nTruth (5 programmes): " + "; ".join(ttl(i) for i in sorted(truth)))
            say("\nYour calculation: ranks of the hits = ____ ; precision at each = ____ ; AP@5 = ____")
            say(SEP + f"hits at ranks {[k for k, i in enumerate(lists[u], 1) if i in truth]}; AP@5 = {ap_at_k(lists[u], truth):.4f}\n")
        else:
            say("\nSure (always in the truth): " + "; ".join(ttl(i) for i in sure))
            say("Tied for the last place (exactly one of them is in the truth, each with probability 1/2): " + "; ".join(ttl(i) for i in tied))
            say("\nYour calculation: AP@5 if the truth is sure + first tied = ____ ; AP@5 if sure + second tied = ____ ; expected AP@5 = their mean = ____")
            aps = [ap_at_k(lists[u], set(sure) | {x}) for x in tied]
            say(SEP + f"AP@5 for the two truths: {aps[0]:.4f} and {aps[1]:.4f}; expected AP@5 = {np.mean(aps):.4f}\n")


if __name__ == "__main__":
    which = sys.argv[1:] or ["screen", "truth", "habit", "ap"]
    say("# Manual checks (numbers for the student; every blank is filled in by the student)\n")
    sp = Split("val12") if any(w in which for w in ("truth", "habit")) else None
    u = None
    if "screen" in which:
        check_screen()
    if "truth" in which:
        u = check_truth(sp)
    if "habit" in which:
        check_habit(sp, u or pick_truth_user(sp)[0])
    if "ap" in which:
        check_ap(sp)
    (ROOT / "MANUAL_CHECKS_data.md").write_text("\n".join(out), encoding="utf-8")
