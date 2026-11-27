"""
Shared pieces of the upgraded pipeline: the data, the split, the model, the search.

The notebooks stay as the record of how the project was first built. These scripts redo the
modeling half so that every number can be rebuilt with one command and nothing leaks.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "heart_disease.csv"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DATA_SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"

RANDOM_STATE = 42                      # the notebooks' seed
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
NUMERICAL = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERICAL + CATEGORICAL     # the order the model is trained and served in

# The search the notebooks ran: 40 random draws, 5 folds, scored by AUC.
PARAM_DIST = {
    "n_estimators": randint(100, 500), "max_depth": randint(3, 10), "learning_rate": uniform(0.01, 0.3),
    "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4), "gamma": uniform(0, 5),
    "reg_alpha": uniform(0, 1), "reg_lambda": uniform(0, 1),
}
N_ITER, FOLDS = 40, 5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_heart():
    """The 303 Cleveland patients. Returns the features in serving order and the target (1 = disease)."""
    if not RAW.exists():
        raise SystemExit("The data file is missing. Run: python pipeline/00_fetch_data.py")
    df = pd.read_csv(RAW, names=COLUMNS, na_values="?")
    y = (df["target"] > 0).astype(int)
    return df[FEATURES], y


def split(X, y):
    """The notebooks' split: 242 patients to train on, 61 held back, same seed."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def imputer():
    """Median for the measurements, most frequent value for the coded fields. Fitted inside whatever it is part of."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    return ColumnTransformer([("num", SimpleImputer(strategy="median"), NUMERICAL),
                              ("cat", SimpleImputer(strategy="most_frequent"), CATEGORICAL)])


def classifier(**params):
    import xgboost as xgb
    return xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1, **params)


def pipeline(resample, **params):
    """Impute, then (if asked) SMOTE, then the model, as one object.
    Inside cross-validation every step is fitted on the training folds only, so no synthetic
    patient is ever a neighbour of a patient the model is then scored on."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    steps = [("impute", imputer())]
    if resample:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", classifier(**params)))
    return Pipeline(steps)


def search(estimator, prefix=""):
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    grid = {prefix + k: v for k, v in PARAM_DIST.items()}
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(estimator, grid, n_iter=N_ITER, cv=folds, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1)


def bootstrap_interval(function, *arrays, draws=2000, seed=RANDOM_STATE):
    """A 95% interval by resampling patients with replacement."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    values = []
    for _ in range(draws):
        rows = rng.integers(0, len(arrays[0]), len(arrays[0]))
        try:
            values.append(function(*[a[rows] for a in arrays]))
        except ValueError:                     # a resample with one class only
            continue
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def exact_interval(successes, n):
    """The exact (Clopper-Pearson) 95% interval for a share, which stays honest at 28 of 28."""
    from scipy.stats import beta
    if n == 0:
        return 0.0, 1.0
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, n - successes + 1))
    high = 1.0 if successes == n else float(beta.ppf(0.975, successes + 1, n - successes))
    return low, high


def save_json(name, data):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")


def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))
