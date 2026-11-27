# Model card: heart disease risk model, version xgb_v2

Built November 23, 2026. A portfolio project on a public research file. It is not a medical device, it
has not been validated for clinical use, and it must not be used on real patients.

## What it is

An XGBoost classifier that takes 13 clinical inputs and returns the probability that a patient has heart
disease, as found by angiography. The API also returns each input's SHAP contribution to that
probability. `GET /data-dictionary` describes the inputs. `GET /model` returns this card's numbers.

## The data

UCI Heart Disease, Cleveland file: 303 patients referred for angiography at one clinic, published in
1988. 139 had disease. 206 are men and 97 are women. Six rows have a missing value. The file's
fingerprint (SHA-256) is checked by `pipeline/00_fetch_data.py` and stored in the model's metadata.

242 patients train the model (111 with disease). 61 are held back (28 with disease).

## How it was trained

Missing values are filled (median for measurements, most frequent code for coded inputs), SMOTE adds
synthetic patients to the smaller class, and XGBoost is fitted. All three steps are one pipeline, so in
cross-validation each is fitted on the training folds only. A random search tries 40 settings over 5
folds. The settings that won are in `models/model_metadata.json`.

## How well it does

| Measure | Value | From |
|---|---|---|
| AUC, nested cross-validation | 0.886 (folds from 0.815 to 0.927) | 15 outer folds on the 242 |
| AUC, test | 0.946 (95% interval 0.884 to 0.990) | the 61, scored once |
| Recall at the served threshold | 0.96 (0.91 to 0.99): 107 of 111 | out-of-fold, the 242 |
| Precision at the served threshold | 0.59 (0.52 to 0.66): 107 of 181 flagged | out-of-fold, the 242 |
| Test, at the served threshold | 28 of 28 caught, 19 false alarms among 33 without disease | the 61, scored once |
| Brier score | 0.141 out-of-fold, 0.100 on the 61 | |

The number to quote is 0.886. The test set gives 0.946, and its interval runs from 0.88 to 0.99: with 61
patients it cannot say more than that. The 61 are an easier draw than the average fold.

The served threshold is 0.136. It comes from counting one missed patient as 3 false alarms, which is a
placeholder until the owner decides (`DECISIONS.md`, decision 1). At that threshold three of every four
patients are flagged.

## By sex

Out-of-fold, at the served threshold:

| | Patients | With disease | Caught | False alarms among those without disease | AUC |
|---|---|---|---|---|---|
| Women | 77 | 18 | 17 | 29 of 59 | 0.842 |
| Men | 165 | 93 | 90 | 45 of 72 | 0.856 |

Only 18 women with disease are in the training data. The interval on 17 of 18 runs from 0.73 to 1.00.
This file cannot show that the model works equally well for women.

## What carries the model

Share of the mean absolute SHAP value on the training patients: thallium test result (thal) 25%, vessels
colored (ca) 20%, chest pain type (cp) 18%. Those three hold 63%.

## Limits

- 303 patients from one clinic, collected decades ago, all of them already referred for angiography. The
  model has never seen a patient who was not suspected of heart disease.
- Every input is required. The model was trained with six filled-in values and is served with none.
- The scores were measured on the same kind of patient the model was trained on. Nothing here measures
  how it does on another hospital's patients.
- The explanation says what moved the model's score. It does not say what causes disease.

## Monitoring

Every scored request is logged: the 13 inputs, the score, the flag, the model version, the time. No name
and no identifier is asked for or stored. `GET /monitoring/drift` compares the last 30, 50 or 100
requests with the training patients. A GitHub Actions job calls the live service each week, checks 20
known patients against their recorded scores, and reads the drift check.

The drift check was tried on replayed patients, not on live traffic: there is none. On the free hosting
plan the log is cleared whenever the service restarts.

## Retraining rules

These rules are written down. None has ever fired, and no outcome labels arrive at this API.

Retrain when any of these holds:

1. The weekly check reports a drift alert two weeks running, and the cause is a real change in who
   arrives or how an input is measured, not a fault upstream. A fault upstream is fixed upstream.
2. Outcomes become available, and on 100 or more labeled patients recall at the served threshold is
   under 0.91, the low end of the interval measured here.
3. An input gains a code the data dictionary does not hold, or a unit changes.
4. A year has passed since the last training.

Retraining means: run `pipeline/00` to `05`, in order, on the new data. Run the tests. Compare the new
nested AUC with 0.886: a change inside the fold spread (0.03) is not a change. Make decision 1 again on
the new table. Give the model a new version name, so the old file stays in the repository and going back
is one line in the metadata.

Do not retrain when the drift check shows "watch" once, or when one week's alert is not repeated.
