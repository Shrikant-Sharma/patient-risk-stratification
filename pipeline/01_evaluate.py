"""
Step 1. Score the model three ways, and say which score can be trusted.

  A. The notebooks' way. SMOTE is applied to the training set once, and the search is then
     cross-validated on that enlarged set. A synthetic patient built from a real one can sit in
     the training folds while the real one is scored. It is run twice: with the notebooks' own
     folds (taken in file order), and with the shuffled folds that B and C use, so that the
     effect of the leak can be read off against B.
  B. SMOTE inside the folds. The same search, with imputing and SMOTE fitted on the training
     folds only.
  C. No resampling. The classes are 139 to 164 in the full file, so there is little to balance.

For B and C the search itself is also put inside an outer cross-validation (nested), repeated
three times, so that the score is of the whole recipe and not of the one best draw.
The 61 held-back patients are scored once for each, with a bootstrap interval.

Usage:  python pipeline/01_evaluate.py            about 3 minutes on 2 cores
        python pipeline/01_evaluate.py --quick    8 draws, 1 repeat: a check that it runs
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as c

QUICK = "--quick" in sys.argv
if QUICK:
    c.N_ITER = 8
REPEATS = 1 if QUICK else 3


def old_way(X_train, y_train, X_test, y_test, notebook_folds):
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RandomizedSearchCV
    imp = c.imputer().fit(X_train)
    Xs, ys = SMOTE(random_state=c.RANDOM_STATE).fit_resample(imp.transform(X_train), y_train)
    if notebook_folds:                     # cv=5 as the notebook wrote it: five folds in file order, the synthetic rows last
        s = RandomizedSearchCV(c.classifier(), c.PARAM_DIST, n_iter=c.N_ITER, cv=c.FOLDS, scoring="roc_auc", random_state=c.RANDOM_STATE, n_jobs=-1).fit(Xs, ys)
    else:
        s = c.search(c.classifier()).fit(Xs, ys)
    p = s.predict_proba(imp.transform(X_test))[:, 1]
    return {"rows_before_smote": int(len(y_train)), "rows_after_smote": int(len(ys)),
            "cv_auc": s.best_score_, "test_auc": roc_auc_score(y_test, p),
            "test_auc_interval": c.bootstrap_interval(roc_auc_score, y_test, p), "best_params": s.best_params_}


def honest(resample, X_train, y_train, X_test, y_test):
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    s = c.search(c.pipeline(resample), prefix="model__").fit(X_train, y_train)
    p = s.predict_proba(X_test)[:, 1]
    out = {"cv_auc_of_best_draw": s.best_score_, "test_auc": roc_auc_score(y_test, p),
           "test_auc_interval": c.bootstrap_interval(roc_auc_score, y_test, p),
           "best_params": {k.replace("model__", ""): v for k, v in s.best_params_.items()}}
    # nested: the search runs again inside each outer training part
    fold_scores, pooled, oof_first = [], [], None
    for repeat in range(REPEATS):
        outer = StratifiedKFold(n_splits=c.FOLDS, shuffle=True, random_state=100 + repeat)
        oof = np.zeros(len(y_train))
        for tr, te in outer.split(X_train, y_train):
            inner = clone(c.search(c.pipeline(resample), prefix="model__")).fit(X_train.iloc[tr], y_train.iloc[tr])
            oof[te] = inner.predict_proba(X_train.iloc[te])[:, 1]
            fold_scores.append(roc_auc_score(y_train.iloc[te], oof[te]))
        pooled.append(roc_auc_score(y_train, oof))
        if repeat == 0:
            oof_first = oof
        print(f"   repeat {repeat + 1} of {REPEATS}: AUC over all out-of-fold patients {pooled[-1]:.3f}", flush=True)
    out.update({"nested_auc_mean_of_folds": float(np.mean(fold_scores)), "nested_auc_sd_of_folds": float(np.std(fold_scores, ddof=1)),
                "nested_auc_lowest_fold": float(np.min(fold_scores)), "nested_auc_highest_fold": float(np.max(fold_scores)),
                "nested_auc_pooled_by_repeat": pooled, "outer_folds_scored": len(fold_scores)})
    return out, oof_first, p


if __name__ == "__main__":
    t0 = time.time()
    X, y = c.load_heart()
    X_train, X_test, y_train, y_test = c.split(X, y)
    report = {"quick": QUICK, "draws": c.N_ITER, "folds": c.FOLDS, "repeats": REPEATS,
              "patients": int(len(y)), "with_disease": int(y.sum()), "train": int(len(y_train)), "train_with_disease": int(y_train.sum()),
              "test": int(len(y_test)), "test_with_disease": int(y_test.sum()),
              "rows_with_a_missing_value": int(X.isna().any(axis=1).sum())}
    print("A. the notebooks' way", flush=True)
    report["A_notebooks_way"] = old_way(X_train, y_train, X_test, y_test, notebook_folds=True)
    report["A_notebooks_way_shuffled_folds"] = old_way(X_train, y_train, X_test, y_test, notebook_folds=False)
    for key, label in (("A_notebooks_way", "the notebooks' folds"), ("A_notebooks_way_shuffled_folds", "shuffled folds")):
        print(f"   {label}: cross-validated AUC {report[key]['cv_auc']:.3f}, test AUC {report[key]['test_auc']:.3f}", flush=True)
    oofs, tests = {}, {}
    for key, label, resample in [("B_smote_inside_folds", "B. SMOTE inside the folds", True), ("C_no_resampling", "C. no resampling", False)]:
        print(label, flush=True)
        report[key], oofs[key], tests[key] = honest(resample, X_train, y_train, X_test, y_test)
        r = report[key]
        print(f"   nested AUC {r['nested_auc_mean_of_folds']:.3f} (sd of folds {r['nested_auc_sd_of_folds']:.3f}), test AUC {r['test_auc']:.3f}", flush=True)

    # The rule, fixed before the numbers: the simpler recipe (C) serves unless B is ahead by more
    # than one standard error of the fold scores.
    b, cc = report["B_smote_inside_folds"], report["C_no_resampling"]
    se = max(b["nested_auc_sd_of_folds"], cc["nested_auc_sd_of_folds"]) / np.sqrt(b["outer_folds_scored"])
    chosen = "B_smote_inside_folds" if b["nested_auc_mean_of_folds"] - cc["nested_auc_mean_of_folds"] > se else "C_no_resampling"
    report["standard_error_used"] = float(se)
    report["chosen"] = chosen
    report["seconds"] = round(time.time() - t0)
    suffix = "_quick" if QUICK else ""
    c.save_json(f"evaluation{suffix}.json", report)
    pd.DataFrame({"row": X_train.index, "y": y_train.values, "B_smote_inside_folds": oofs["B_smote_inside_folds"],
                  "C_no_resampling": oofs["C_no_resampling"]}).to_csv(c.RESULTS / f"out_of_fold_predictions{suffix}.csv", index=False)
    pd.DataFrame({"row": X_test.index, "y": y_test.values, "B_smote_inside_folds": tests["B_smote_inside_folds"],
                  "C_no_resampling": tests["C_no_resampling"]}).to_csv(c.RESULTS / f"test_predictions{suffix}.csv", index=False)
    print(f"chosen: {chosen}. {report['seconds']} seconds. results/evaluation{suffix}.json")
