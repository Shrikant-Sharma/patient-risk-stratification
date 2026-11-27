"""
Close-out check (November 2026): the seed rule.

Two sentences in this project quote a gain of one trained recipe over another:
  1. SMOTE inside the folds against no resampling (nested AUC).
  2. The leak: the notebooks' way on shuffled folds against SMOTE inside the folds (the search's best draw).
Each rested on one seed. This script trains both sides again on five seeds and reports each gain
as a range. The seed sets the model's sampling, SMOTE, the search's draws and the search's folds.
The split into 242 and 61 patients and the outer folds stay as they are, so only the training moves.

    python pipeline/09_seed_check.py           about 8 minutes
    python pipeline/09_seed_check.py --quick   a check that it runs

Writes results/seed_check.json.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as c

spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).parent / "01_evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)                 # reads --quick from the command line, as step 01 does
SEEDS = [42, 7, 11, 23, 101]


if __name__ == "__main__":
    t0 = time.time()
    X, y = c.load_heart()
    X_train, X_test, y_train, y_test = c.split(X, y)          # the split is made once, with the project's seed
    rows = []
    for seed in SEEDS[:2] if evaluate.QUICK else SEEDS:
        c.RANDOM_STATE = seed
        leak = evaluate.old_way(X_train, y_train, X_test, y_test, notebook_folds=False)
        b, _, _ = evaluate.honest(True, X_train, y_train, X_test, y_test)
        cc, _, _ = evaluate.honest(False, X_train, y_train, X_test, y_test)
        se = max(b["nested_auc_sd_of_folds"], cc["nested_auc_sd_of_folds"]) / np.sqrt(b["outer_folds_scored"])
        rows.append({"seed": seed, "nested_with_smote": b["nested_auc_mean_of_folds"], "nested_without": cc["nested_auc_mean_of_folds"],
                     "smote_gain": b["nested_auc_mean_of_folds"] - cc["nested_auc_mean_of_folds"], "standard_error": float(se),
                     "smote_chosen_by_the_rule": bool(b["nested_auc_mean_of_folds"] - cc["nested_auc_mean_of_folds"] > se),
                     "best_draw_with_leak": leak["cv_auc"], "best_draw_without_leak": b["cv_auc_of_best_draw"],
                     "leak": leak["cv_auc"] - b["cv_auc_of_best_draw"], "test_auc_with_smote": b["test_auc"]})
        r = rows[-1]
        print(f"seed {seed:>3}: nested AUC {r['nested_with_smote']:.3f} with SMOTE, {r['nested_without']:.3f} without, gain {r['smote_gain']:+.3f} "
              f"(one standard error {r['standard_error']:.3f}). leak {r['leak']:+.3f}", flush=True)
    c.RANDOM_STATE = 42

    def span(key):
        v = [r[key] for r in rows]
        return {"low": float(min(v)), "high": float(max(v)), "mean": float(np.mean(v))}

    report = {"quick": evaluate.QUICK, "seeds": [r["seed"] for r in rows], "by_seed": rows,
              "smote_gain": span("smote_gain"), "leak": span("leak"), "nested_with_smote": span("nested_with_smote"),
              "nested_without": span("nested_without"), "test_auc_with_smote": span("test_auc_with_smote"),
              "seeds_where_the_rule_chose_smote": int(sum(r["smote_chosen_by_the_rule"] for r in rows)), "seconds": round(time.time() - t0)}
    c.save_json("seed_check_quick.json" if evaluate.QUICK else "seed_check.json", report)
    g, k = report["smote_gain"], report["leak"]
    print(f"SMOTE's gain over {len(rows)} seeds: {g['low']:+.3f} to {g['high']:+.3f}. The rule chose SMOTE in {report['seeds_where_the_rule_chose_smote']} of {len(rows)}.")
    print(f"the leak over {len(rows)} seeds: {k['low']:+.3f} to {k['high']:+.3f}")
    print(f"nested AUC with SMOTE: {report['nested_with_smote']['low']:.3f} to {report['nested_with_smote']['high']:.3f}. {report['seconds']} seconds.")
