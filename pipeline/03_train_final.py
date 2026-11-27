"""
Step 3. Train the model that is served, and write everything the API needs beside it.

The recipe chosen in step 1 is fitted on the 242 training patients. Two files come out:

  models/xgb_v2.json          the trees, in XGBoost's own format
  models/model_metadata.json  the feature order, the threshold, the calibration as numbers,
                              the reference the drift check compares against, the measured
                              scores, the library versions and the data file's fingerprint

Nothing is pickled. A pickle loads only under the library versions that wrote it, and loading
one can run code. The first version needed a pickled SHAP explainer that was not in the
repository, so a fresh clone could not start the API.

Usage:  python pipeline/03_train_final.py
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import common as c
from src import data_dictionary, monitoring

VERSION = "xgb_v2"

if __name__ == "__main__":
    import imblearn, sklearn, xgboost
    ev, tc = c.read_json("evaluation.json"), c.read_json("threshold_calibration.json")
    chosen = ev["chosen"]
    X, y = c.load_heart()
    X_train, X_test, y_train, y_test = c.split(X, y)
    fitted = c.search(c.pipeline(chosen == "B_smote_inside_folds"), prefix="model__").fit(X_train, y_train).best_estimator_
    model = fitted.named_steps["model"]
    c.MODELS.mkdir(exist_ok=True)
    model.get_booster().save_model(str(c.MODELS / f"{VERSION}.json"))

    # the same 61 scores as step 1, or something is not reproducible
    p_test = fitted.predict_proba(X_test)[:, 1]
    saved = pd.read_csv(c.RESULTS / "test_predictions.csv")[chosen].values
    assert np.allclose(p_test, saved, atol=1e-6), "the final model does not reproduce step 1's test scores"

    imp = fitted.named_steps["impute"]
    train_imputed = pd.DataFrame(imp.transform(X_train), columns=c.FEATURES)
    oof = pd.read_csv(c.RESULTS / "out_of_fold_predictions.csv")[chosen].values
    columns = {name: train_imputed[name].values for name in c.FEATURES}
    columns[monitoring.SCORE] = oof
    kinds = {name: data_dictionary.FIELDS[name]["kind"] for name in c.FEATURES}
    kinds[monitoring.SCORE] = "number"
    reference = monitoring.build_reference(columns, kinds, seed=c.RANDOM_STATE)

    # how the served threshold treats women and men, on out-of-fold scores, and which inputs carry the model
    flagged, sex, yt = oof >= tc["threshold"], train_imputed["sex"].values, y_train.values
    by_sex = {}
    for code, label in ((0, "women"), (1, "men")):
        m = sex == code
        caught, sick = int((flagged & m & (yt == 1)).sum()), int((m & (yt == 1)).sum())
        alarms, well = int((flagged & m & (yt == 0)).sum()), int((m & (yt == 0)).sum())
        from sklearn.metrics import roc_auc_score
        by_sex[label] = {"patients": int(m.sum()), "with_disease": sick, "caught": caught, "recall": caught / sick, "recall_interval": c.exact_interval(caught, sick),
                         "false_alarms": alarms, "without_disease": well, "auc": float(roc_auc_score(yt[m], oof[m]))}
    import xgboost as xgb
    contribs = model.get_booster().predict(xgb.DMatrix(train_imputed.to_numpy(dtype=np.float32), feature_names=c.FEATURES), pred_contribs=True)[:, :-1]
    weight = np.abs(contribs).mean(axis=0)
    importance = {n: float(w / weight.sum()) for n, w in sorted(zip(c.FEATURES, weight), key=lambda t: -t[1])}

    upper = next(r["threshold"] for r in tc["threshold_table_out_of_fold"] if r["ratio"] == 1)
    meta = {
        "model_version": VERSION, "built": time.strftime("%Y-%m-%d", time.gmtime()), "model_file": f"{VERSION}.json",
        "features": c.FEATURES,
        "recipe": {"resampling": "SMOTE inside each training fold" if chosen == "B_smote_inside_folds" else "none",
                   "search": f"{c.N_ITER} random draws, {c.FOLDS} folds, scored by AUC", "parameters": ev[chosen]["best_params"],
                   "trained_on": int(len(y_train)), "seed": c.RANDOM_STATE},
        "fill_in_for_missing": {**{n: float(v) for n, v in zip(c.NUMERICAL, imp.named_transformers_["num"].statistics_)},
                                **{n: float(v) for n, v in zip(c.CATEGORICAL, imp.named_transformers_["cat"].statistics_)}},
        "calibration": tc["calibration_served"],
        "threshold": tc["threshold"], "cost_of_a_miss_against_a_false_alarm": tc["cost_ratio_used"],
        "tiers": {"LOW": f"below the threshold ({tc['threshold']:.3f})", "MODERATE": f"from the threshold up to {upper:.3f}",
                  "HIGH": f"{upper:.3f} and above, where a patient is flagged even if a miss and a false alarm cost the same", "upper_cut": upper},
        "measured": {
            "auc_nested_cross_validation": ev[chosen]["nested_auc_mean_of_folds"], "auc_lowest_fold": ev[chosen]["nested_auc_lowest_fold"],
            "auc_highest_fold": ev[chosen]["nested_auc_highest_fold"], "auc_test": ev[chosen]["test_auc"], "auc_test_interval": ev[chosen]["test_auc_interval"],
            "at_threshold_out_of_fold": tc["at_threshold_out_of_fold"], "at_threshold_test": tc["test_once"]["at_threshold"],
            "test_recall_interval": tc["test_once"]["recall_interval"], "brier_out_of_fold": tc["calibration_out_of_fold"][tc["calibration_chosen"]]["brier"],
            "by_sex_out_of_fold": by_sex, "share_of_mean_absolute_shap": importance},
        "training_range": {n: {"lowest": float(train_imputed[n].min()), "median": float(train_imputed[n].median()), "highest": float(train_imputed[n].max())} for n in c.NUMERICAL},
        "drift_reference": reference,
        "data": {"file": "data/raw/heart_disease.csv", "sha256": c.sha256(c.RAW), "patients": int(len(y))},
        "versions": {"python": platform.python_version(), "xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__,
                     "imbalanced-learn": imblearn.__version__, "numpy": np.__version__},
    }
    (c.MODELS / "model_metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # 20 test patients with the score the served model must give them: the live check and the tests use these
    known = [{"features": {k: float(v) for k, v in X_test.iloc[i].items()}, "raw_score": float(p_test[i])}
             for i in range(len(X_test)) if not X_test.iloc[i].isna().any()][:20]
    (c.MODELS / "known_answers.json").write_text(json.dumps(known, indent=1), encoding="utf-8")
    print(f"models/{VERSION}.json and models/model_metadata.json written. threshold {meta['threshold']:.3f}, calibration {meta['calibration']['method']}")
    print("chance of a false drift alert in one window when nothing has changed:",
          ", ".join(f"{w} requests {reference['rule_false_alert_rate'][str(w)]:.1%}" for w in monitoring.WINDOWS))
    print("by sex, out of fold:", {k: (v["caught"], v["with_disease"], v["false_alarms"], v["without_disease"], round(v["auc"], 3)) for k, v in by_sex.items()})
    print("share of mean |SHAP|:", {k: round(v, 3) for k, v in list(importance.items())[:5]})
    print("chance of a watch:", ", ".join(f"{w} requests {reference['watch_rate_when_nothing_changed'][str(w)]:.1%}" for w in monitoring.WINDOWS))
    print("limits at 50 requests:", {n: round(f["limits"]["50"], 3) for n, f in reference["features"].items()})
