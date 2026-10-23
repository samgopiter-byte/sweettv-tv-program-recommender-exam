"""How many 'new' truth programmes are new episodes / parts of a series the user already committed to?"""
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import lite
from common import PROC

PREFIX = re.compile(r"^(т/с|х/ф|м/с|д/с|м/ф|д/ф|т/ш|ток-шоу|шоу)\s+", re.I)


def series_key(title):
    t = str(title).strip()
    m = re.search(r"['\"«“]([^'\"»”]{2,})['\"»”]", t)
    if m:
        t = m.group(1)
    else:
        t = re.split(r",|\d+\s*(с\.|эп|ч\.|сезон)", t)[0]
    t = PREFIX.sub("", t).strip(" .,;:-").lower()
    return t


if __name__ == "__main__":
    from common import PROC
    sch = pd.read_parquet(os.path.join(PROC, "schedule.parquet"), columns=["tv_show_id", "tv_show_title"])
    sch = sch[sch["tv_show_id"] != 0]
    title = sch.groupby("tv_show_id")["tv_show_title"].agg(lambda s: s.value_counts().index[0])
    key = {int(i): series_key(t) for i, t in title.items()}
    n_ids, n_keys = len(key), len(set(key.values()))
    print(f"programme ids {n_ids}, distinct series keys {n_keys} (ids per key {n_ids / n_keys:.2f})")
    ex = pd.Series(key).groupby(pd.Series(key)).size().sort_values(ascending=False).head(6)
    print("keys with most ids:", dict(ex))
    L = lite.Lite("v6")
    out = []
    for mo in lite.FOLDS:
        me = L.meta[mo]
        items = me["items"]
        pos = {u: i for i, u in enumerate(me["users"])}
        t_new = same_series = known_cand = 0
        for u in me["eligible"]:
            i = pos[u]
            hist_ids = {int(items[j]) for j in me["hist_items"][i]}
            hist_keys = {key.get(x) for x in hist_ids}
            sure, tied, need, det = me["info"][u]
            for it in set(sure) | set(tied):
                if it in hist_ids:
                    continue
                t_new += 1
                same_series += key.get(it) in hist_keys
        out.append((mo, t_new, same_series / max(t_new, 1)))
        print(f"{mo}: new truth items {t_new}; share whose series key was committed before: {same_series / max(t_new, 1):.3f}", flush=True)
