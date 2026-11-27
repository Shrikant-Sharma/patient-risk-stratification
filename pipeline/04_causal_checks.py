"""
Step 4. Run the refutation checks on the estimator the headline number comes from.

The causal notebook reports one number: quitting smoking raised weight by 3.27 kg over ten
years among the quitters (NHEFS data, 1,566 people). That number comes from the notebook's
own matching: a logistic propensity model with squared terms, then one non-quitter for each
quitter, nearest on the logit of the propensity score, within 0.2 standard deviations, each
non-quitter used once.

The notebook's three refutation checks were run on a different estimator: DoWhy's default
propensity matching, which gave 2.04 kg. They said nothing about the 3.27.

This script wraps the headline estimator in one function and runs the checks on it:

  placebo        shuffle who quit. The estimate should fall to about zero.
  random cause   add a made-up confounder. The estimate should hardly move.
  subset         keep a random 80% of people. The estimate should stay close.
  caliper        match within 0.1, 0.2 and 0.3 standard deviations.
  bootstrap      resample people, then fit and match again each time, so the interval
                 carries the uncertainty of the matching and not only of the pairs.

Usage:  python pipeline/04_causal_checks.py          about a minute
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as c

SIMULATIONS = 200
BOOTSTRAPS = 1000


def load_nhefs():
    from causaldata import nhefs
    df = nhefs.load_pandas().data.dropna(subset=["wt82_71"]).copy()
    X = pd.DataFrame({"sex": df["sex"].astype(int), "race": df["race"].astype(int), "age": df["age"], "age_sq": df["age"] ** 2,
                      "smokeintensity": df["smokeintensity"], "smokeintensity_sq": df["smokeintensity"] ** 2,
                      "smokeyrs": df["smokeyrs"], "smokeyrs_sq": df["smokeyrs"] ** 2, "wt71": df["wt71"], "wt71_sq": df["wt71"] ** 2,
                      "exercise_num": df["exercise"].astype(int), "active_num": df["active"].astype(int)})
    X = pd.concat([X, pd.get_dummies(df["education"].astype(int), prefix="edu", drop_first=True).astype(int)], axis=1)
    return X.to_numpy(dtype=float), df["qsmk"].astype(int).to_numpy(), df["wt82_71"].to_numpy(dtype=float), list(X.columns)


def estimate(X, treated, outcome, caliper_sd=0.2, details=False):
    """The headline estimator, start to finish: propensity model, matching, mean difference of the pairs."""
    from scipy.spatial.distance import cdist
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    keep = X.std(axis=0) > 0                                  # a resample can lose a rare education level
    Xs = StandardScaler().fit_transform(X[:, keep])
    ps = LogisticRegression(penalty=None, max_iter=5000, solver="lbfgs").fit(Xs, treated).predict_proba(Xs)[:, 1]
    ps = np.clip(ps, 1e-10, 1 - 1e-10)
    lp = np.log(ps / (1 - ps))
    caliper = caliper_sd * lp.std(ddof=1)
    t, k = np.where(treated == 1)[0], np.where(treated == 0)[0]
    dist = cdist(lp[t].reshape(-1, 1), lp[k].reshape(-1, 1))
    used, pairs = np.zeros(len(k), dtype=bool), []
    for i in np.argsort(-np.abs(lp[t])):                      # the hardest to match go first, as in the notebook
        row = np.where(used, np.inf, dist[i])
        j = int(np.argmin(row))
        if row[j] <= caliper:
            pairs.append((t[i], k[j]))
            used[j] = True
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    att = float(np.mean(outcome[a] - outcome[b]))
    if not details:
        return att
    pooled = np.sqrt((X[treated == 1].var(axis=0, ddof=1) + X[treated == 0].var(axis=0, ddof=1)) / 2)
    smd_before = (X[treated == 1].mean(axis=0) - X[treated == 0].mean(axis=0)) / pooled
    smd_after = (X[a].mean(axis=0) - X[b].mean(axis=0)) / pooled
    return {"att": att, "pairs": len(pairs), "quitters": int(len(t)), "diffs": outcome[a] - outcome[b],
            "largest_gap_before": float(np.max(np.abs(smd_before))), "largest_gap_after": float(np.max(np.abs(smd_after)))}


def summary(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5)), "runs": int(len(values))}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    X, treated, outcome, names = load_nhefs()
    rng = np.random.default_rng(c.RANDOM_STATE)
    n = len(treated)
    base = estimate(X, treated, outcome, details=True)
    diffs = base.pop("diffs")
    report = {"people": n, "naive_difference": float(outcome[treated == 1].mean() - outcome[treated == 0].mean()), "headline": base}
    pair_boot = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(5000)]
    report["interval_pairs_only"] = [float(np.percentile(pair_boot, 2.5)), float(np.percentile(pair_boot, 97.5))]
    print(f"headline estimator: {base['att']:+.3f} kg from {base['pairs']} pairs of {base['quitters']} quitters", flush=True)

    placebo = np.array([estimate(X, rng.permutation(treated), outcome) for _ in range(SIMULATIONS)])
    report["placebo"] = {**summary(placebo), "share_as_large_as_headline": float(np.mean(np.abs(placebo) >= abs(base["att"])))}
    report["random_common_cause"] = summary([estimate(np.column_stack([X, rng.normal(size=n)]), treated, outcome) for _ in range(SIMULATIONS)])
    subset = []
    for _ in range(SIMULATIONS):
        rows = rng.choice(n, int(0.8 * n), replace=False)
        subset.append(estimate(X[rows], treated[rows], outcome[rows]))
    report["subset_80_percent"] = summary(subset)
    report["caliper"] = {}
    for width in (0.1, 0.2, 0.3):
        d = estimate(X, treated, outcome, caliper_sd=width, details=True)
        report["caliper"][str(width)] = {"att": d["att"], "pairs": d["pairs"], "largest_gap_after": d["largest_gap_after"]}
    boot = []
    for _ in range(BOOTSTRAPS):
        rows = rng.integers(0, n, n)
        boot.append(estimate(X[rows], treated[rows], outcome[rows]))
    report["bootstrap_whole_procedure"] = summary(boot)
    report["notebook_refuters_were_run_on"] = {"estimator": "DoWhy backdoor.propensity_score_matching, default settings", "estimate": 2.042}
    c.save_json("causal_checks.json", report)

    r = report
    print(f"  balance: largest standardized gap {base['largest_gap_before']:.2f} before matching, {base['largest_gap_after']:.2f} after")
    print(f"  placebo:        mean {r['placebo']['mean']:+.3f} kg, 95% of runs {r['placebo']['low']:+.2f} to {r['placebo']['high']:+.2f}, runs as large as the headline: {r['placebo']['share_as_large_as_headline']:.0%}")
    print(f"  random cause:   mean {r['random_common_cause']['mean']:+.3f} kg, 95% of runs {r['random_common_cause']['low']:+.2f} to {r['random_common_cause']['high']:+.2f}")
    print(f"  80% subset:     mean {r['subset_80_percent']['mean']:+.3f} kg, 95% of runs {r['subset_80_percent']['low']:+.2f} to {r['subset_80_percent']['high']:+.2f}")
    print("  caliper:        " + ", ".join(f"{k} SD {v['att']:+.2f} kg ({v['pairs']} pairs)" for k, v in r["caliper"].items()))
    print(f"  interval, pairs only:       {r['interval_pairs_only'][0]:+.2f} to {r['interval_pairs_only'][1]:+.2f} kg")
    print(f"  interval, whole procedure:  {r['bootstrap_whole_procedure']['low']:+.2f} to {r['bootstrap_whole_procedure']['high']:+.2f} kg (mean {r['bootstrap_whole_procedure']['mean']:+.2f})")
