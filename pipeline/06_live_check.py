"""
Step 6. Check a running API from outside: is it up, does it still give the recorded answers,
and what does its drift check say?

It uses only Python's standard library, so the weekly GitHub Actions job installs nothing.
A free Render service sleeps after 15 minutes without traffic and takes about a minute to
wake, so the first call waits and retries.

Usage:  python pipeline/06_live_check.py https://your-service.onrender.com
        python pipeline/06_live_check.py http://127.0.0.1:8000
Exit code 0 when every check passes, 1 when one fails.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOLERANCE = 1e-4


def call(url, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "X-Check": "weekly"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read())


def wait_until_up(base, patience=240):
    start = time.time()
    while True:
        try:
            return call(base + "/health"), time.time() - start
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if time.time() - start > patience:
                raise SystemExit(f"FAIL  no answer from {base}/health in {patience} seconds: {error}")
            time.sleep(10)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Give the service's address, for example: python pipeline/06_live_check.py http://127.0.0.1:8000")
    base = sys.argv[1].rstrip("/")
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    known = json.loads((ROOT / "models" / "known_answers.json").read_text(encoding="utf-8"))
    failures = []
    health, waited = wait_until_up(base)
    print(f"up after {waited:.0f} seconds: {health}")
    if health.get("model_version") != meta["model_version"]:
        failures.append(f"the service runs {health.get('model_version')}, the repository holds {meta['model_version']}")
    worst, start = 0.0, time.time()
    for item in known:
        got = call(base + "/predict", item["features"])
        worst = max(worst, abs(got["disease_probability"] - item["raw_score"])) if meta["calibration"]["method"] == "raw" else worst
        if got["prediction"] != int(got["disease_probability"] >= meta["threshold"]) or got["threshold_used"] != round(meta["threshold"], 4):
            failures.append(f"threshold or flag differs for a known patient: {got}")
    seconds = (time.time() - start) / len(known)
    print(f"{len(known)} known patients: largest difference from the recorded score {worst:.6f}, {seconds:.2f} seconds a request")
    if worst > TOLERANCE:
        failures.append(f"a known patient's score differs from the recorded one by {worst:.6f}")
    explained = call(base + "/explain", known[0]["features"])
    if len(explained["contributions"]) != len(meta["features"]):
        failures.append("the explanation does not cover every input")
    dictionary = call(base + "/data-dictionary")
    if list(dictionary["fields"]) != meta["features"]:
        failures.append("the data dictionary does not match the model's inputs")
    drift = call(base + "/monitoring/drift")
    print("drift check:", drift.get("status"), "| requests in the window:", drift.get("requests_logged"),
          "| inputs over their limit:", drift.get("inputs_over_limit", []))
    if drift.get("status") == "alert":
        failures.append(f"drift alert: inputs {drift.get('inputs_over_limit')}, score over its limit: {drift.get('score_over_limit')}")
    for line in failures:
        print("FAIL ", line)
    print("all checks passed" if not failures else f"{len(failures)} check(s) failed")
    raise SystemExit(1 if failures else 0)
