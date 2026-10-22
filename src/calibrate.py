"""Does the local replica of the Kaggle score reproduce what Kaggle returned for our three submissions?

The three submitted models (v1, blend_v2, ltr_v2) are rebuilt for the 12-week fold val12 (history 8 weeks, target 12
weeks; the LTR trained on the labels of the folds val and val2) and scored with the tie-aware replica. They are compared
with the scores Kaggle returned for the same three models on the real test window.
"""
import hs
import methods as m
import rerank as rr
import final as f1
import final2 as f2

KAGGLE = {"v1": (0.28940, 0.26088), "blend_v2": (0.30515, 0.27332), "ltr_v2": (0.32193, 0.29401)}

if __name__ == "__main__":
    sp, M, fb, D = rr.get_D("val12")
    info = hs.tie_info(hs.qualifying_counts(sp.inter, sp.tgt_lo, sp.tgt_hi))
    samples = hs.truth_samples(info)
    users = [u for u in sp.users if u in samples]
    print("val12: eligible template users", len(users))

    lists = {}
    lists["popularity"] = {u: fb[:5] for u in sp.users}
    lists["v1"] = m.recommend(M, f1.final_scores(M), fb)
    names = list(D["names"])
    lists["blend_v2"] = rr.lists_from_scores(sp, M, fb, D["rows"], D["cols"], D["X"][:, names.index("blend")])
    Ds = [rr.get_D("val")[3], rr.get_D("val2")[3]]
    model, idx, _ = f2.train_final(Ds)
    lists["ltr_v2"] = rr.lists_from_scores(sp, M, fb, D["rows"], D["cols"], model.predict(D["X"][:, idx]))
    print(f"{'model':12s} {'replica MAP@5 (expected)':>26s} {'(earliest-first)':>17s} | Kaggle public / private")
    for name, L in lists.items():
        r = hs.evaluate_samples(L, samples, users, info)
        k = KAGGLE.get(name)
        print(f"{name:12s} {r['MAP@5']:26.4f} {r['MAP@5_det']:17.4f} | " + (f"{k[0]:.5f} / {k[1]:.5f}" if k else "-"))
