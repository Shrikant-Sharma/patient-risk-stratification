"""
The request log and the drift check. Plain numpy and the standard library, so the API can run it.

Request log: one JSON line for each scored request: time, the 13 inputs, the score, the flag,
the model version. No name and no identifier is asked for or stored.

Drift check: for each input and for the score, compare the last N requests with the training
patients, using the population stability index (PSI). The usual fixed limits (0.1 and 0.25)
were written for samples of thousands. At 50 requests PSI is well above 0.1 by chance alone,
so each limit here is measured: the value that samples of N training patients stay under 99
times in 100, and a second, strict limit that they stay under 999 times in 1,000.

The rule. Alert: the score is over its limit, or two or more inputs are over theirs, or one
input is over its strict limit. Watch: one input is over its limit and under the strict one.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

WINDOWS = [30, 50, 100]
PERCENTILE = 99
STRICT_PERCENTILE = 99.9
SCORE = "score"


def log_path():
    return Path(os.environ.get("REQUEST_LOG", Path(__file__).resolve().parent.parent / "logs" / "requests.jsonl"))


def log_request(features, raw_score, probability, flagged, model_version, endpoint):
    """Append one line. A log that cannot be written must never fail a prediction."""
    line = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "endpoint": endpoint, "model_version": model_version,
            "features": features, "raw_score": round(float(raw_score), 6), "probability": round(float(probability), 6), "flagged": int(flagged)}
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    except OSError:
        pass
    return line


def read_log(last=None):
    path = log_path()
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-last:] if last else rows


# ---- the reference: how the training patients are spread over a few bins ----

def bin_spec(values, kind):
    values = np.asarray(values, dtype=float)
    if kind == "code":
        return {"kind": "code", "codes": sorted(float(v) for v in np.unique(values))}
    edges = np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8]))
    return {"kind": "number", "edges": [float(e) for e in edges]}


def counts(values, spec):
    values = np.asarray(values, dtype=float)
    if spec["kind"] == "code":
        c = [int(np.sum(values == code)) for code in spec["codes"]]
        return np.array(c + [len(values) - sum(c)])            # last bin: a code never seen in training
    return np.bincount(np.searchsorted(spec["edges"], values, side="right"), minlength=len(spec["edges"]) + 1)


def psi(reference_counts, current_counts):
    """Population stability index. Half a patient is added to every bin so an empty bin does not give infinity."""
    r = np.asarray(reference_counts, dtype=float) + 0.5
    k = np.asarray(current_counts, dtype=float) + 0.5
    r, k = r / r.sum(), k / k.sum()
    return float(np.sum((k - r) * np.log(k / r)))


def build_reference(columns, kinds, seed=42, draws=5000):
    """columns: name -> training values (the score included). Returns bins, counts and the measured limits."""
    rng = np.random.default_rng(seed)
    names = list(columns)
    n = len(next(iter(columns.values())))
    ref = {"patients": n, "percentile": PERCENTILE, "strict_percentile": STRICT_PERCENTILE, "draws": draws, "features": {}, "rule_false_alert_rate": {}}
    for name in names:
        spec = bin_spec(columns[name], kinds[name])
        ref["features"][name] = {**spec, "counts": [int(v) for v in counts(columns[name], spec)], "limits": {}, "strict_limits": {}}
    for window in WINDOWS:
        values = {name: np.zeros(draws) for name in names}
        for d in range(draws):
            rows = rng.integers(0, n, window)
            for name in names:
                f = ref["features"][name]
                values[name][d] = psi(f["counts"], counts(np.asarray(columns[name])[rows], f))
        for name in names:
            ref["features"][name]["limits"][str(window)] = float(np.percentile(values[name], PERCENTILE))
            ref["features"][name]["strict_limits"][str(window)] = float(np.percentile(values[name], STRICT_PERCENTILE))
        over = np.array([values[name] > ref["features"][name]["limits"][str(window)] for name in names])
        strict = np.array([values[name] > ref["features"][name]["strict_limits"][str(window)] for name in names])
        inputs = [i for i, name in enumerate(names) if name != SCORE]
        alert = over[names.index(SCORE)] | (over[inputs].sum(axis=0) >= 2) | strict[inputs].any(axis=0)
        ref["rule_false_alert_rate"][str(window)] = float(alert.mean())
        ref.setdefault("watch_rate_when_nothing_changed", {})[str(window)] = float((over[inputs].any(axis=0) & ~alert).mean())
    return ref


def window_for(n):
    usable = [w for w in WINDOWS if w <= n]
    return max(usable) if usable else None


def drift_report(rows, reference):
    """rows: log lines, oldest first. Checks the latest window the log can fill.
    Alert: the score is over its limit, or two or more inputs are, or one input is over its strict limit.
    Watch: one input is over its limit and under the strict one."""
    window = window_for(len(rows))
    if window is None:
        return {"status": "not enough requests", "requests_logged": len(rows), "needed": min(WINDOWS)}
    rows = rows[-window:]
    out = {"window": window, "requests_logged": len(rows), "features": {}}
    for name, f in reference["features"].items():
        values = [r["raw_score"] if name == SCORE else r["features"][name] for r in rows]
        value, limit, strict = psi(f["counts"], counts(values, f)), f["limits"][str(window)], f["strict_limits"][str(window)]
        out["features"][name] = {"psi": round(value, 4), "limit": round(limit, 4), "strict_limit": round(strict, 4),
                                 "over": bool(value > limit), "over_strict": bool(value > strict)}
    moved = [n for n, v in out["features"].items() if v["over"] and n != SCORE]
    far = [n for n in moved if out["features"][n]["over_strict"]]
    out["inputs_over_limit"], out["inputs_over_strict_limit"] = moved, far
    out["score_over_limit"] = out["features"][SCORE]["over"]
    out["status"] = "alert" if out["score_over_limit"] or len(moved) >= 2 or far else ("watch" if moved else "ok")
    out["flag_rate"] = round(float(np.mean([r["flagged"] for r in rows])), 4)
    return out
