"""
Step 2. Pick the calibration and the threshold on patients the model did not train on.

The first version picked its threshold on the 61 test patients and then reported recall and
precision on the same 61. Here both choices are made on out-of-fold predictions for the 242
training patients (each scored by a model that never saw that patient), and the 61 are looked
at once, at the end.

Calibration, three ways: the raw score, Platt scaling (a logistic curve fitted to the score),
and isotonic regression (a step curve). Each calibrator is itself cross-fitted, so it is never
scored on the patients it was fitted to. The rule, fixed before the numbers: the raw score
serves unless a calibrator lowers the Brier score by more than chance, meaning the 95%
bootstrap interval of the difference lies below zero.

Threshold: from what a miss costs against a false alarm. That ratio is a decision, not a
result. It is read from decision.json, and the table shows what each ratio would do.

Usage:  python pipeline/02_threshold_and_calibration.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as c

RATIOS = [1, 2, 3, 5, 10]
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return {"method": "platt", "a": float(m.coef_[0, 0]), "b": float(m.intercept_[0])}


def fit_isotonic(p, y):
    from sklearn.isotonic import IsotonicRegression
    m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
    return {"method": "isotonic", "x": [float(v) for v in m.X_thresholds_], "y": [float(v) for v in m.y_thresholds_]}


def apply(cal, p):
    """The same arithmetic the API runs: plain numpy, no fitted object."""
    p = np.asarray(p, dtype=float)
    if cal["method"] == "raw":
        return p
    if cal["method"] == "platt":
        return 1 / (1 + np.exp(-(cal["a"] * logit(p) + cal["b"])))
    return np.interp(p, cal["x"], cal["y"])


FITTERS = {"raw": lambda p, y: {"method": "raw"}, "platt": fit_platt, "isotonic": fit_isotonic}


def cross_fitted(method, p, y):
    from sklearn.model_selection import StratifiedKFold
    out = np.zeros(len(p))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=c.RANDOM_STATE).split(p, y):
        out[te] = apply(FITTERS[method](p[tr], y[tr]), p[te])
    return out


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def gap(y, p, bins=5):
    """Expected calibration error on equal-sized bins: the average distance between what was predicted and what happened."""
    order = np.argsort(p)
    parts = np.array_split(order, bins)
    return float(sum(len(k) * abs(np.mean(p[k]) - np.mean(y[k])) for k in parts) / len(p))


def at(threshold, y, p, ratio):
    flag = p >= threshold
    tp, fp, fn, tn = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()), int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    return {"threshold": float(threshold), "flagged": tp + fp, "caught": tp, "missed": fn, "false_alarms": fp, "cleared_correctly": tn,
            "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1), "cost": ratio * fn + fp}


def best_threshold(y, p, ratio):
    """The lowest-cost cut on these patients. Ties go to the cut nearest the one theory gives, 1 / (1 + ratio)."""
    cuts = np.unique(np.round(np.concatenate([p, [0.5]]), 4))
    costs = np.array([at(t, y, p, ratio)["cost"] for t in cuts])
    ties = cuts[costs == costs.min()]
    return float(ties[np.argmin(np.abs(ties - 1 / (1 + ratio)))])


if __name__ == "__main__":
    ev = c.read_json("evaluation.json")
    chosen = ev["chosen"]
    oof = pd.read_csv(c.RESULTS / "out_of_fold_predictions.csv")
    test = pd.read_csv(c.RESULTS / "test_predictions.csv")
    y, p = oof["y"].values, oof[chosen].values
    yt, pt = test["y"].values, test[chosen].values
    decision = json.loads((c.ROOT / "decision.json").read_text(encoding="utf-8"))
    ratio = decision["cost_of_a_miss_against_a_false_alarm"]
    report = {"recipe": chosen, "out_of_fold_patients": int(len(y)), "test_patients": int(len(yt)), "cost_ratio_used": ratio}

    # calibration
    cross = {m: cross_fitted(m, p, y) for m in FITTERS}
    report["calibration_out_of_fold"] = {}
    for m in FITTERS:
        row = {"brier": brier(y, cross[m]), "gap": gap(y, cross[m])}
        if m != "raw":
            row["brier_minus_raw_interval"] = c.bootstrap_interval(lambda a, b, r: brier(a, b) - brier(a, r), y, cross[m], cross["raw"])
        report["calibration_out_of_fold"][m] = row
    better = [m for m in ("platt", "isotonic") if report["calibration_out_of_fold"][m]["brier_minus_raw_interval"][1] < 0]
    method = min(better, key=lambda m: report["calibration_out_of_fold"][m]["brier"]) if better else "raw"
    report["calibration_chosen"] = method
    fitted = {m: FITTERS[m](p, y) for m in FITTERS}          # each fitted on all 242, for serving and for the test look
    report["calibration_served"] = fitted[method]
    report["calibration_fitted_all"] = fitted

    # threshold, on the served probability
    served_oof = cross[method]
    report["threshold_table_out_of_fold"] = []
    for r in RATIOS:
        t = best_threshold(y, served_oof, r)
        report["threshold_table_out_of_fold"].append({"ratio": r, "theory": 1 / (1 + r), **at(t, y, served_oof, r)})
    threshold = best_threshold(y, served_oof, ratio)
    report["threshold"] = threshold
    report["at_threshold_out_of_fold"] = at(threshold, y, served_oof, ratio)
    def intervals(row):
        return {"recall_interval": c.exact_interval(row["caught"], row["caught"] + row["missed"]),
                "precision_interval": c.exact_interval(row["caught"], row["flagged"])}
    report["at_threshold_out_of_fold"].update(intervals(report["at_threshold_out_of_fold"]))
    report["half_point_out_of_fold"] = at(0.5, y, served_oof, ratio)

    # the one look at the 61
    served_test = apply(fitted[method], pt)
    test_row = at(threshold, yt, served_test, ratio)
    report["test_once"] = {"at_threshold": test_row, **intervals(test_row),
                           "calibration": {m: {"brier": brier(yt, apply(fitted[m], pt)), "gap": gap(yt, apply(fitted[m], pt))} for m in FITTERS}}
    # the first version's threshold, for the record: picked on the 61, here scored on the 242
    report["first_version_threshold_out_of_fold"] = at(0.4226, y, p, ratio)
    c.save_json("threshold_calibration.json", report)

    print(f"recipe: {chosen}. calibration chosen: {method}")
    for m, row in report["calibration_out_of_fold"].items():
        extra = "" if m == "raw" else "  difference from raw, 95% interval {:+.4f} to {:+.4f}".format(*row["brier_minus_raw_interval"])
        print(f"  {m:9s} Brier {row['brier']:.4f}  gap {row['gap']:.3f}{extra}")
    print("cost ratio  threshold  flagged  caught  missed  false alarms  recall  precision")
    for row in report["threshold_table_out_of_fold"]:
        print(f"  {row['ratio']:>3} to 1    {row['threshold']:.3f}     {row['flagged']:>4}    {row['caught']:>4}    {row['missed']:>4}      {row['false_alarms']:>4}       {row['recall']:.2f}     {row['precision']:.2f}")
    a, b = report["at_threshold_out_of_fold"], report["test_once"]["at_threshold"]
    print(f"served threshold {threshold:.3f} at {ratio} to 1")
    print(f"  242 out-of-fold: recall {a['recall']:.2f} ({a['recall_interval'][0]:.2f} to {a['recall_interval'][1]:.2f}), precision {a['precision']:.2f}")
    ri = report["test_once"]["recall_interval"]
    print(f"  61 test, once:   caught {b['caught']} of {b['caught'] + b['missed']} (the true rate could be as low as {ri[0]:.2f}), false alarms {b['false_alarms']}, precision {b['precision']:.2f}")
