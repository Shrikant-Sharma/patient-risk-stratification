"""
Run every step that needs no key, no account and no decision, in order, and keep a log.

    python run_all.py                      every step
    python run_all.py --from 02            start at step 02, for example after a change to decision.json
    python run_all.py --only 03,05a,tests  only these steps
    python run_all.py --list               show the steps and stop

Each command runs with the Python of this folder's .venv, whichever Python started this script.
The script prints each step's name and how long it took, stops at the first failure and shows
the error, and saves everything it prints to results/run_log.txt. It also writes
results/run_environment.json: the date, the machine and the package versions of the run.

It uses only Python's standard library.
"""
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
LOG = ROOT / "results" / "run_log.txt"
ENVIRONMENT = ROOT / "results" / "run_environment.json"

# name, what it does, and the command handed to the .venv's Python ("api" is the function below)
STEPS = [
    ("00", "fetch the public data file and check its fingerprint", ["pipeline/00_fetch_data.py"]),
    ("01", "score the model three ways (the long step)", ["pipeline/01_evaluate.py"]),
    ("02", "calibration and threshold on out-of-fold scores", ["pipeline/02_threshold_and_calibration.py"]),
    ("03", "train the served model and write its metadata", ["pipeline/03_train_final.py"]),
    ("04", "causal checks on the headline estimator", ["pipeline/04_causal_checks.py"]),
    ("05a", "drift check: replay four patient streams", ["pipeline/05_drift_monitor.py", "--replay"]),
    ("05b", "drift check: compare four alert rules", ["pipeline/05_drift_monitor.py", "--limits"]),
    ("06", "the API on this machine: start it, live check, one bad request, stop it", "api"),
    ("tests", "the offline checks, on the files this run wrote", ["-m", "pytest", "-q"]),
]
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "shap", "numba", "causaldata",
            "fastapi", "uvicorn", "pydantic", "httpx", "pytest"]
EXAMPLE = {"age": 60, "sex": 1, "cp": 4, "trestbps": 140, "chol": 280, "fbs": 0, "restecg": 2,
           "thalach": 130, "exang": 1, "oldpeak": 2.5, "slope": 2, "ca": 2, "thal": 7}
CHILD = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

log_file = None


def say(text=""):
    """Print a line and write the same line to the log."""
    print(text, flush=True)
    if log_file:
        log_file.write(text + "\n")
        log_file.flush()


def run(command):
    """Run one command with the .venv's Python. Its output goes to the screen and the log, line by line."""
    process = subprocess.Popen([str(PY)] + command, cwd=ROOT, env=CHILD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")

    def copy():
        for line in process.stdout:
            say("   " + line.rstrip())

    reader = threading.Thread(target=copy, daemon=True)
    reader.start()
    code = process.wait()
    reader.join(timeout=15)             # do not hang if a worker process still holds the pipe
    return code


def call(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "X-Check": "weekly"})
    try:
        with urllib.request.urlopen(request, timeout=30) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as refusal:          # a refusal is an answer too
        return refusal.code, json.loads(refusal.read())


def free_port(preferred=8000):
    for port in (preferred, 0):                        # 0 asks the system for any free port
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise SystemExit("no free port on this machine")


def api():
    """Start the API on a local port, check it from outside, send one request it must refuse, stop it."""
    base = f"http://127.0.0.1:{free_port()}"
    handle, name = tempfile.mkstemp(suffix=".txt")
    os.close(handle)
    with open(name, "w+", encoding="utf-8", errors="replace") as output:
        server = subprocess.Popen([str(PY), "-m", "uvicorn", "src.app:app", "--host", "127.0.0.1", "--port", base.rsplit(":", 1)[1]],
                                  cwd=ROOT, env=CHILD, stdout=output, stderr=subprocess.STDOUT)
        try:
            say(f"   uvicorn src.app:app at {base}")
            started, up = time.time(), False
            while time.time() - started < 90 and server.poll() is None:
                try:
                    up = call(base + "/health")[0] == 200
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    time.sleep(0.5)
            if not up:
                say("   the API did not start. Its own output follows.")
                return 1
            say(f"   up after {time.time() - started:.1f} seconds")
            code = run(["pipeline/06_live_check.py", base])
            if code:
                return code
            status, body = call(base + "/predict", {**EXAMPLE, "chol": 5.2, "thal": 5})
            say(f"   one bad request (cholesterol 5.2, thal 5): HTTP {status}")
            detail = body.get("detail")
            for item in detail if isinstance(detail, list) else []:
                say(f"      refused: {item['loc'][-1]}: {item['msg']}")
            if status != 422:
                say("   expected a refusal (HTTP 422)")
                return 1
            status, model = call(base + "/model")
            say(f"   /model: {model['model_version']}, built {model['built']}, threshold {model['threshold']}, "
                f"a miss counted as {model['cost_of_a_miss_against_a_false_alarm']} false alarms")
            return 0
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
            output.seek(0)
            for line in output.read().splitlines():
                say("   [server] " + line)
    try:
        os.remove(name)
    except OSError:
        pass


def cpu_name():
    try:
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment():
    """The machine and the versions this run used. The computer's name is left out on purpose."""
    code = ("import json, platform, importlib.metadata as m\n"
            f"names = {PACKAGES!r}\n"
            "out = {'python': platform.python_version(), 'packages': {}}\n"
            "for n in names:\n"
            "    try:\n"
            "        out['packages'][n] = m.version(n)\n"
            "    except m.PackageNotFoundError:\n"
            "        out['packages'][n] = 'not installed'\n"
            "print(json.dumps(out))")
    found = json.loads(subprocess.run([str(PY), "-c", code], capture_output=True, text=True, check=True).stdout)
    return {"system": platform.platform(), "processor": cpu_name(), "logical_cores": os.cpu_count(),
            "python": found["python"], "packages": found["packages"],
            "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "commit": git("rev-parse", "--short", "HEAD")}}


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    global log_file
    args = sys.argv[1:]
    names = [s[0] for s in STEPS]
    if "--list" in args:
        for name, what, _ in STEPS:
            print(f"{name:6s} {what}")
        return 0
    chosen = STEPS
    if "--from" in args:
        start = args[args.index("--from") + 1]
        if start not in names:
            raise SystemExit(f"No step named {start}. The steps are: {', '.join(names)}")
        chosen = STEPS[names.index(start):]
    if "--only" in args:
        wanted = args[args.index("--only") + 1].split(",")
        unknown = [w for w in wanted if w not in names]
        if unknown:
            raise SystemExit(f"No step named {', '.join(unknown)}. The steps are: {', '.join(names)}")
        chosen = [s for s in STEPS if s[0] in wanted]
    if not PY.exists():
        raise SystemExit(f"No .venv in {ROOT}. Create it and install requirements-pipeline.txt first (see the README).")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    whole = len(chosen) == len(STEPS)
    LOG.parent.mkdir(exist_ok=True)
    log_file = open(LOG, "w" if whole else "a", encoding="utf-8")       # a part run adds to the log of the whole run
    env = environment()
    record = {"started": now(), "steps_asked_for": [s[0] for s in chosen], "steps": []}
    say("=" * 78)
    say(f"run_all.py  started {record['started']}  steps: {', '.join(record['steps_asked_for'])}")
    say(f"machine: {env['system']}, {env['processor']}, {env['logical_cores']} logical cores")
    say(f"python {env['python']} from .venv  |  branch {env['git']['branch']} at {env['git']['commit']}")
    say("packages: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items()))
    say("=" * 78)

    t0, failed = time.time(), None
    for name, what, command in chosen:
        say()
        say(f"[{name}] {what}")
        start = time.time()
        code = api() if command == "api" else run(command)
        seconds = round(time.time() - start, 1)
        record["steps"].append({"step": name, "seconds": seconds, "passed": code == 0})
        if code != 0:
            failed = name
            say(f"[{name}] FAILED after {seconds} seconds (exit code {code})")
            break
        say(f"[{name}] done in {seconds} seconds")

    record.update({"finished": now(), "seconds": round(time.time() - t0, 1), "all_passed": failed is None})
    say()
    say("step    seconds  result")
    for s in record["steps"]:
        say(f"{s['step']:6s} {s['seconds']:8.1f}  {'ok' if s['passed'] else 'FAILED'}")
    if failed:
        say(f"STOPPED at step {failed}. The error is above. No later step was run.")
    else:
        say(f"all {len(record['steps'])} steps passed in {record['seconds']:.0f} seconds")
    say("log: results/run_log.txt")

    earlier = []
    if not whole and ENVIRONMENT.exists():
        earlier = json.loads(ENVIRONMENT.read_text(encoding="utf-8")).get("runs", [])
    ENVIRONMENT.write_text(json.dumps({**env, "runs": earlier + [record]}, indent=1), encoding="utf-8")
    log_file.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
