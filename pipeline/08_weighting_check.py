"""
Close-out check (November 2026): is weighting steadier than matching when a noise variable is added?

Step 04 showed that a made-up confounder moves the matched estimate by up to half a kilogram,
because greedy one-to-one matching re-pairs people. This script puts inverse probability weighting
beside it: the same propensity model, the same people, the same made-up confounder in each run.
Both estimate the effect among the quitters.

    python pipeline/08_weighting_check.py          about a minute

Writes results/weighting_check.json.
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as c

spec = importlib.util.spec_from_file_location("causal", Path(__file__).parent / "04_causal_checks.py")
causal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(causal)
SIMULATIONS = 200


def weighted(X, treated, outcome):
    """Inverse probability weighting for the quitters: the same propensity model as the matching."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    keep = X.std(axis=0) > 0
    Xs = StandardScaler().fit_transform(X[:, keep])
    ps = LogisticRegression(penalty=None, max_iter=5000, solver="lbfgs").fit(Xs, treated).predict_proba(Xs)[:, 1]
    ps = np.clip(ps, 1e-10, 1 - 1e-10)
    w = ps[treated == 0] / (1 - ps[treated == 0])             # a non-quitter counts by how much they look like a quitter
    return float(outcome[treated == 1].mean() - np.sum(w * outcome[treated == 0]) / np.sum(w))


def spread(values):
    v = np.asarray(values)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5)),
            "width": float(np.percentile(v, 97.5) - np.percentile(v, 2.5))}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    X, treated, outcome, _ = causal.load_nhefs()
    n = len(treated)
    rng = np.random.default_rng(c.RANDOM_STATE)
    report = {"people": n, "simulations": SIMULATIONS,
              "estimate_on_all_people": {"matching": causal.estimate(X, treated, outcome), "weighting": weighted(X, treated, outcome)}}
    runs = {"random_common_cause": {"matching": [], "weighting": []}, "subset_80_percent": {"matching": [], "weighting": []},
            "placebo": {"matching": [], "weighting": []}}
    for _ in range(SIMULATIONS):
        noisy = np.column_stack([X, rng.normal(size=n)])       # the same made-up confounder for both estimators
        runs["random_common_cause"]["matching"].append(causal.estimate(noisy, treated, outcome))
        runs["random_common_cause"]["weighting"].append(weighted(noisy, treated, outcome))
        rows = rng.choice(n, int(0.8 * n), replace=False)      # the same 80% of people for both
        runs["subset_80_percent"]["matching"].append(causal.estimate(X[rows], treated[rows], outcome[rows]))
        runs["subset_80_percent"]["weighting"].append(weighted(X[rows], treated[rows], outcome[rows]))
        shuffled = rng.permutation(treated)                    # the same shuffle for both
        runs["placebo"]["matching"].append(causal.estimate(X, shuffled, outcome))
        runs["placebo"]["weighting"].append(weighted(X, shuffled, outcome))
    e = report["estimate_on_all_people"]
    print(f"on all {n} people: matching {e['matching']:+.3f} kg, weighting {e['weighting']:+.3f} kg")
    print("check                  estimator   mean     sd    middle 95% of 200 runs   width")
    for check, both in runs.items():
        report[check] = {name: spread(values) for name, values in both.items()}
        for name in ("matching", "weighting"):
            s = report[check][name]
            print(f"{check:22s} {name:10s} {s['mean']:+.2f}  {s['sd']:.3f}   {s['low']:+.2f} to {s['high']:+.2f}          {s['width']:.2f}")
        report[check]["sd_ratio_weighting_over_matching"] = report[check]["weighting"]["sd"] / report[check]["matching"]["sd"]
    c.save_json("weighting_check.json", report)
    print("results/weighting_check.json")
