"""Build and cache the re-ranker features for the extra folds (val12, val3, val5)."""
import sys
import time
import rerank as rr

if __name__ == "__main__":
    for mode in sys.argv[1:] or ["val12", "val3", "val5"]:
        t = time.time()
        sp, M, fb, D = rr.get_D(mode)
        n_truth = sum(len(sp.truth[u]) for u in sp.users if u in sp.truth)
        print(f"{mode}: rows {len(D['rows'])}, positives {int(D['y'].sum())} of {n_truth} truth items, "
              f"features {len(D['names'])}, {time.time() - t:.0f}s", flush=True)
