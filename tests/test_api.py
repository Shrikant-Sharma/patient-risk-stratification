"""
Checks that need no network and no account. They run on every push (.github/workflows/tests.yml).

    python -m pytest -q
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["REQUEST_LOG"] = str(Path(tempfile.mkdtemp()) / "requests.jsonl")

from fastapi.testclient import TestClient  # noqa: E402

from src import data_dictionary, monitoring  # noqa: E402
from src.app import app, calibrate  # noqa: E402

META = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
KNOWN = json.loads((ROOT / "models" / "known_answers.json").read_text(encoding="utf-8"))
EXAMPLE = {"age": 60, "sex": 1, "cp": 4, "trestbps": 140, "chol": 280, "fbs": 0, "restecg": 2,
           "thalach": 130, "exang": 1, "oldpeak": 2.5, "slope": 2, "ca": 2, "thal": 7}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_a_fresh_clone_has_every_file_the_api_loads():
    assert (ROOT / "models" / META["model_file"]).exists()
    assert not list((ROOT / "src").glob("*.pkl"))
    text = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
    assert "joblib" not in text and "pickle" not in text


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["model_version"] == META["model_version"]


def test_known_patients_get_the_scores_the_build_recorded(client):
    assert len(KNOWN) == 20
    for item in KNOWN:
        got = client.post("/predict", json=item["features"]).json()
        assert abs(got["disease_probability"] - calibrate(item["raw_score"], META["calibration"])) < 1e-4
        assert got["prediction"] == int(calibrate(item["raw_score"], META["calibration"]) >= META["threshold"])


def test_feature_order_is_the_metadata_order_not_the_request_order(client):
    shuffled = dict(reversed(list(EXAMPLE.items())))
    assert client.post("/predict", json=shuffled).json() == client.post("/predict", json=EXAMPLE).json()


def test_tiers_agree_with_the_flag(client):
    for item in KNOWN:
        got = client.post("/predict", json=item["features"]).json()
        assert (got["risk_tier"] == "LOW") == (got["prediction"] == 0)


def test_explanation_adds_up_to_the_score(client):
    body = client.post("/explain", json=EXAMPLE).json()
    total = body["base_value_log_odds"] + sum(c["shap_value"] for c in body["contributions"])
    raw = 1 / (1 + np.exp(-total))
    assert len(body["contributions"]) == 13
    assert abs(calibrate(raw, META["calibration"]) - body["disease_probability"]) < 2e-3


def test_explanation_matches_the_shap_library():
    shap = pytest.importorskip("shap")
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(ROOT / "models" / META["model_file"]))
    X = np.array([[item["features"][n] for n in META["features"]] for item in KNOWN], dtype=np.float32)
    ours = booster.predict(xgb.DMatrix(X, feature_names=META["features"]), pred_contribs=True)
    theirs = shap.TreeExplainer(booster).shap_values(X)
    assert np.allclose(ours[:, :-1], theirs, atol=1e-4)


@pytest.mark.parametrize("field,value", [("thal", 5), ("cp", 0), ("ca", 4), ("sex", 2), ("chol", 5.2), ("age", 150), ("slope", 0)])
def test_values_outside_the_dictionary_are_refused(client, field, value):
    reply = client.post("/predict", json={**EXAMPLE, field: value})
    assert reply.status_code == 422


def test_a_missing_field_is_refused(client):
    body = dict(EXAMPLE)
    del body["thal"]
    assert client.post("/predict", json=body).status_code == 422


def test_data_dictionary_covers_every_model_input(client):
    body = client.get("/data-dictionary").json()
    assert list(body["fields"]) == META["features"]
    for name, f in body["fields"].items():
        assert f["meaning"] and (f["kind"] == "code" and f["values"] or f["unit"])
    assert body["fields"]["chol"]["seen_in_training"]["median"] > 0
    assert set(data_dictionary.FIELDS) == set(META["features"])


def test_model_facts_carry_the_measured_scores(client):
    body = client.get("/model").json()
    assert body["threshold"] == META["threshold"] and "drift_reference" not in body
    assert 0.5 < body["measured"]["auc_nested_cross_validation"] < body["measured"]["auc_test"] <= 1


def test_every_scored_request_is_logged_without_identifiers(client):
    before = len(monitoring.read_log())
    client.post("/predict", json=EXAMPLE)
    client.post("/explain", json=EXAMPLE)
    client.post("/predict", json={**EXAMPLE, "thal": 5})          # refused, so not logged
    rows = monitoring.read_log()
    assert len(rows) == before + 2
    assert set(rows[-1]) == {"time", "endpoint", "model_version", "features", "raw_score", "probability", "flagged"}
    assert set(rows[-1]["features"]) == set(META["features"])


def test_the_weekly_check_marks_its_requests_and_they_are_not_logged(client):
    before = len(monitoring.read_log())
    assert client.post("/predict", json=EXAMPLE, headers={"X-Check": "weekly"}).status_code == 200
    assert len(monitoring.read_log()) == before


def test_a_log_that_cannot_be_written_does_not_fail_the_prediction(client, monkeypatch):
    monkeypatch.setenv("REQUEST_LOG", str(Path(__file__) / "requests.jsonl"))   # under a file: no folder can be made there, on any system
    assert client.post("/predict", json=EXAMPLE).status_code == 200


def test_psi_is_zero_for_the_same_spread_and_grows_with_a_shift():
    ref = [50, 50, 50, 50, 50]
    assert monitoring.psi(ref, ref) == pytest.approx(0, abs=1e-9)
    assert monitoring.psi(ref, [10, 10, 10, 10, 10]) < 0.01
    assert monitoring.psi(ref, [0, 0, 10, 20, 20]) > monitoring.psi(ref, [6, 8, 10, 12, 14]) > 0
    assert np.isfinite(monitoring.psi(ref, [50, 0, 0, 0, 0]))


def test_bins_put_an_unseen_code_in_its_own_bin():
    spec = monitoring.bin_spec([3, 3, 6, 7, 7], "code")
    assert list(monitoring.counts([3, 7, 5], spec)) == [1, 0, 1, 1]
    spec = monitoring.bin_spec(np.arange(100), "number")
    assert monitoring.counts(np.arange(100), spec).sum() == 100 and len(spec["edges"]) == 4


def _rows(n, shift=0.0):
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        item = KNOWN[rng.integers(0, len(KNOWN))]
        rows.append({"features": {**item["features"], "age": item["features"]["age"] + shift}, "raw_score": item["raw_score"], "flagged": 1})
    return rows


def test_drift_report_needs_30_requests_and_uses_the_largest_window_it_can_fill():
    ref = META["drift_reference"]
    assert monitoring.drift_report(_rows(12), ref)["status"] == "not enough requests"
    assert monitoring.drift_report(_rows(45), ref)["window"] == 30
    assert monitoring.drift_report(_rows(260), ref)["window"] == 100


def test_drift_report_sees_every_patient_30_years_older():
    report = monitoring.drift_report(_rows(100, shift=30), META["drift_reference"])
    assert report["features"]["age"]["over"] and "age" in report["inputs_over_strict_limit"]
    assert report["status"] == "alert"


def test_one_input_just_over_its_limit_is_a_watch_and_far_over_is_an_alert():
    ref = json.loads(json.dumps(META["drift_reference"]))
    rows = _rows(100)
    base = monitoring.drift_report(rows, ref)
    for f in ref["features"].values():                      # make every input pass, then move one limit at a time
        f["limits"]["100"], f["strict_limits"]["100"] = 99.0, 999.0
    assert monitoring.drift_report(rows, ref)["status"] == "ok"
    value = base["features"]["chol"]["psi"]
    ref["features"]["chol"]["limits"]["100"] = value / 2
    assert monitoring.drift_report(rows, ref)["status"] == "watch"
    ref["features"]["chol"]["strict_limits"]["100"] = value / 1.5
    assert monitoring.drift_report(rows, ref)["status"] == "alert"


def test_limits_are_measured_and_wider_for_smaller_windows():
    for name, f in META["drift_reference"]["features"].items():
        assert f["limits"]["30"] > f["limits"]["50"] > f["limits"]["100"] > 0, name
        assert all(f["strict_limits"][w] > f["limits"][w] for w in ("30", "50", "100")), name
        assert sum(f["counts"]) == META["drift_reference"]["patients"]
    for rate in META["drift_reference"]["rule_false_alert_rate"].values():
        assert rate < 0.05


def test_calibration_arithmetic():
    assert calibrate(0.3, {"method": "raw"}) == 0.3
    assert calibrate(0.5, {"method": "platt", "a": 1.0, "b": 0.0}) == pytest.approx(0.5)
    assert calibrate(0.5, {"method": "platt", "a": 2.0, "b": 1.0}) == pytest.approx(1 / (1 + np.exp(-1)))
    assert calibrate(0.25, {"method": "isotonic", "x": [0.0, 0.5, 1.0], "y": [0.0, 0.4, 1.0]}) == pytest.approx(0.2)


def test_metadata_names_the_data_file_and_the_decision():
    assert len(META["data"]["sha256"]) == 64 and META["data"]["patients"] == 303
    decision = json.loads((ROOT / "decision.json").read_text(encoding="utf-8"))
    assert decision["cost_of_a_miss_against_a_false_alarm"] == META["cost_of_a_miss_against_a_false_alarm"]


def test_no_secret_and_no_patient_file_is_in_the_repository():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/raw" in ignore and "logs/" in ignore
