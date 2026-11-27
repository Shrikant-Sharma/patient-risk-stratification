"""
My own change (November 26, 2026): how far does the threshold move when the patients change?

Step 02 gives one lowest-cost cut for each cost ratio, from the 242 out-of-fold scores. This script
resamples those 242 patients 2,000 times, with replacement, and finds the lowest-cost cut again each
time. It reports the middle 95% of the cuts for each ratio, and the recall and precision of the served
cut over the same resamples. The 61 held-back patients are not used.

    python pipeline/07_threshold_bootstrap.py

Writes results/threshold_bootstrap.json.
"""
import json

import numpy as np
import pandas as pd

import common as c

DRAWS, SEED, RATIOS = 2000, 42, [1, 2, 3, 5, 10]


def best_cut(y, p, ratio):
    """The same rule as step 02: the lowest-cost cut, ties to the cut nearest 1 / (1 + ratio)."""
    cuts = np.unique(np.round(np.concatenate([p, [0.5]]), 4))
    flag = p[None, :] >= cuts[:, None]
    cost = ratio * (~flag & (y == 1)).sum(axis=1) + (flag & (y == 0)).sum(axis=1)
    ties = cuts[cost == cost.min()]
    return float(ties[np.argmin(np.abs(ties - 1 / (1 + ratio)))])


if __name__ == "__main__":
    step02 = c.read_json("threshold_calibration.json")
    oof = pd.read_csv(c.RESULTS / "out_of_fold_predictions.csv")
    y, p = oof["y"].values, oof[step02["recipe"]].values
    served = step02["threshold"]
    for row in step02["threshold_table_out_of_fold"]:              # the rule here must be step 02's rule
        assert abs(best_cut(y, p, row["ratio"]) - row["threshold"]) < 1e-9, row

    rng = np.random.default_rng(SEED)
    cuts = {r: [] for r in RATIOS}
    recall, precision, flagged = [], [], []
    for _ in range(DRAWS):
        i = rng.integers(0, len(y), len(y))
        yb, pb = y[i], p[i]
        for r in RATIOS:
            cuts[r].append(best_cut(yb, pb, r))
        flag = pb >= served
        recall.append((flag & (yb == 1)).sum() / max((yb == 1).sum(), 1))
        precision.append((flag & (yb == 1)).sum() / max(flag.sum(), 1))
        flagged.append(flag.mean())

    def span(values):
        low, mid, high = np.percentile(values, [2.5, 50, 97.5])
        return {"low": round(float(low), 4), "median": round(float(mid), 4), "high": round(float(high), 4)}

    report = {"draws": DRAWS, "seed": SEED, "patients": int(len(y)), "served_threshold": served,
              "cost_ratio_used": step02["cost_ratio_used"], "cut_by_ratio": {}}
    print(f"{DRAWS} resamples of the {len(y)} out-of-fold patients")
    print("cost ratio  cut on all 242   middle 95% of resampled cuts   median   share of resamples within 0.02 of it")
    for r in RATIOS:
        full = best_cut(y, p, r)
        near = float(np.mean(np.abs(np.array(cuts[r]) - full) <= 0.02))
        report["cut_by_ratio"][str(r)] = {"cut_on_all_patients": full, **span(cuts[r]), "share_within_0.02": round(near, 3)}
        s = report["cut_by_ratio"][str(r)]
        print(f"  {r:>3} to 1      {full:.3f}          {s['low']:.3f} to {s['high']:.3f}              {s['median']:.3f}      {near:.0%}")
    report["at_served_threshold"] = {"recall": span(recall), "precision": span(precision), "share_flagged": span(flagged)}
    a = report["at_served_threshold"]
    print(f"at the served cut {served:.3f}: recall {a['recall']['low']:.2f} to {a['recall']['high']:.2f}, "
          f"precision {a['precision']['low']:.2f} to {a['precision']['high']:.2f}, share flagged {a['share_flagged']['low']:.2f} to {a['share_flagged']['high']:.2f}")
    (c.RESULTS / "threshold_bootstrap.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("results/threshold_bootstrap.json")
