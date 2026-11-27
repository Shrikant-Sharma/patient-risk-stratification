"""
Step 5. Replay patients through the API and see what the drift check says.

No live traffic exists for a portfolio model, so the check is tried on streams where the
answer is known. Every patient is a real row of the public file.

  as published   the 61 held-back patients. Nothing has changed, so no alert should fire.
  older clinic   only patients aged 60 and over. Who arrives has changed.
  cuff reads high  the 61 held-back patients with resting blood pressure 15 mm Hg too high.
                 A measurement has changed.
  no chest pain  only patients with chest pain type 4. A referral pattern has changed.

Each stream goes through the real API code (FastAPI's test client), is written to its own
request log, and the last 50 requests are compared with the training patients.
Then each stream is drawn 200 times at 50 requests, to count how often the alert fires.

--limits compares four alert rules on the same streams, for the decision on the drift limits.

With no argument it checks whatever is in the API's own request log.

Usage:  python pipeline/05_drift_monitor.py --replay
        python pipeline/05_drift_monitor.py --limits
        python pipeline/05_drift_monitor.py              the live log (logs/requests.jsonl)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import common as c
from src import monitoring

WINDOW, DRAWS = 50, 200


def streams(meta):
    X, y = c.load_heart()
    X = X.fillna(meta["fill_in_for_missing"])                # the API takes no missing value, so fill as training did
    X_train, X_test, y_train, y_test = c.split(X, y)
    return {
        "as published": (X_test, y_test, "the 61 held-back patients"),
        "older clinic": (X[X["age"] >= 60], None, "patients aged 60 and over, from the whole file"),
        "cuff reads high": (X_test.assign(trestbps=X_test["trestbps"] + 15), y_test, "the 61 held-back patients, blood pressure 15 mm Hg too high"),
        "no chest pain": (X[X["cp"] == 4], None, "patients with chest pain type 4, from the whole file"),
    }


def through_api(frame, name):
    from fastapi.testclient import TestClient
    log = Path(tempfile.mkdtemp()) / f"{name.replace(' ', '_')}.jsonl"
    os.environ["REQUEST_LOG"] = str(log)
    from src.app import app
    with TestClient(app) as client:
        for _, row in frame.iterrows():
            reply = client.post("/predict", json={k: float(v) for k, v in row.items()})
            assert reply.status_code == 200, reply.text
        report = client.get("/monitoring/drift").json()
    return report, monitoring.read_log()


RULES = {
    "fixed 0.10, any input or the score": lambda f: any(v["psi"] > 0.10 for v in f.values()),
    "fixed 0.25, any input or the score": lambda f: any(v["psi"] > 0.25 for v in f.values()),
    "measured limit, any input or the score": lambda f: any(v["over"] for v in f.values()),
    "as built: score, or two inputs, or one input over its strict limit":
        lambda f: f["score"]["over"] or sum(v["over"] for k, v in f.items() if k != "score") >= 2 or any(v["over_strict"] for k, v in f.items() if k != "score"),
}


def compare_limits(meta):
    """How often each rule alerts at 50 requests: on training patients (nothing changed), and on the four streams."""
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(c.MODELS / meta["model_file"]))
    X, y = c.load_heart()
    X = X.fillna(meta["fill_in_for_missing"])
    X_train = c.split(X, y)[0]
    oof = pd.read_csv(c.RESULTS / "out_of_fold_predictions.csv")[c.read_json("evaluation.json")["chosen"]].values
    sets = {"training patients, nothing changed": (X_train, oof)}
    for name, (frame, _, _) in streams(meta).items():
        sets[name] = (frame, booster.predict(xgb.DMatrix(frame[meta["features"]].to_numpy(dtype=np.float32), feature_names=meta["features"])))
    rng, draws, out = np.random.default_rng(c.RANDOM_STATE + 1), 500, {}
    for name, (frame, scores) in sets.items():
        fired = {rule: 0 for rule in RULES}
        for _ in range(draws):
            pick = rng.choice(len(frame), WINDOW, replace=len(frame) < WINDOW)
            rows = [{"features": frame.iloc[i].to_dict(), "raw_score": float(scores[i]), "flagged": 0} for i in pick]
            features = monitoring.drift_report(rows, meta["drift_reference"])["features"]
            for rule, test in RULES.items():
                fired[rule] += bool(test(features))
        out[name] = {rule: n / draws for rule, n in fired.items()}
        print(f"{name:36s} " + "  ".join(f"{v:5.0%}" for v in out[name].values()), flush=True)
    c.save_json("drift_limits.json", {"window": WINDOW, "draws": draws, "rules": list(RULES), "share_of_draws_with_an_alert": out})


if __name__ == "__main__":
    meta = json.loads((c.MODELS / "model_metadata.json").read_text(encoding="utf-8"))
    reference = meta["drift_reference"]
    if "--limits" in sys.argv:
        compare_limits(meta)
        raise SystemExit
    if "--replay" not in sys.argv:
        rows = monitoring.read_log()
        print(f"{monitoring.log_path()}: {len(rows)} requests")
        print(json.dumps(monitoring.drift_report(rows, reference), indent=1))
        raise SystemExit
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(c.MODELS / meta["model_file"]))
    rng = np.random.default_rng(c.RANDOM_STATE)
    out = {"window": WINDOW, "draws": DRAWS, "streams": {}}
    for name, (frame, labels, what) in streams(meta).items():
        sample = frame if len(frame) <= 61 else frame.sample(61, random_state=c.RANDOM_STATE)
        report, rows = through_api(sample, name)
        entry = {"what": what, "patients_available": int(len(frame)), "sent": int(len(sample)), "status": report["status"],
                 "inputs_over_limit": report["inputs_over_limit"], "inputs_over_strict_limit": report["inputs_over_strict_limit"], "score_over_limit": report["score_over_limit"],
                 "flag_rate": report["flag_rate"], "psi": {k: v["psi"] for k, v in report["features"].items()},
                 "limit": {k: v["limit"] for k, v in report["features"].items()}}
        if labels is not None:
            flagged = np.array([r["flagged"] for r in rows])
            entry["caught"], entry["with_disease"] = int((flagged[labels.values == 1]).sum()), int(labels.sum())
            entry["false_alarms"] = int((flagged[labels.values == 0]).sum())
        # how often does the alert fire at 50 requests?
        scores = booster.predict(xgb.DMatrix(frame[meta["features"]].to_numpy(dtype=np.float32), feature_names=meta["features"]))
        tally = {"alert": 0, "watch": 0, "ok": 0}
        for _ in range(DRAWS):
            pick = rng.choice(len(frame), WINDOW, replace=len(frame) < WINDOW)
            drawn = [{"features": frame.iloc[i].to_dict(), "raw_score": float(scores[i]), "flagged": int(scores[i] >= meta["threshold"])} for i in pick]
            tally[monitoring.drift_report(drawn, reference)["status"]] += 1
        entry["of_200_draws"] = tally
        out["streams"][name] = entry
        print(f"{name:16s} {entry['status']:6s} inputs over their limit: {', '.join(entry['inputs_over_limit']) or 'none'}; score over: {entry['score_over_limit']}; "
              f"flagged {entry['flag_rate']:.0%}; 200 draws: {tally['alert']} alert, {tally['watch']} watch, {tally['ok']} ok", flush=True)
    c.save_json("drift_replay.json", out)
