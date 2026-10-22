"""Build and cache the v4 (truth-aligned) re-ranker data for the given folds."""
import sys
import time
import numpy as np
import rerank2 as r2

if __name__ == "__main__":
    for mode in sys.argv[1:] or ["val12", "val", "val2", "val3", "val5", "test"]:
        t = time.time()
        sp, M, fb, D = r2.get_D2(mode)
        msg = f"{mode}: rows {len(D['rows'])}, features {len(D['names'])}"
        if "elig" in D:
            n_elig = len(np.unique(D["rows"][D["elig"]]))
            msg += f", eligible users {n_elig}, rows with y_bin=1: {int(D['y_bin'].sum())}, mean y_soft mass per eligible user {D['y_soft'][D['elig']].sum() / max(n_elig, 1):.2f}"
        print(msg + f", {time.time() - t:.0f}s", flush=True)
