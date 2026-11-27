# Model card: heart disease risk model, version xgb_v2

Built November 23, 2026. Trained and measured again on my own machine on November 26, 2026: every number
here is from that run. A portfolio project on a public research file. It is not a medical device, it
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
folds. The settings that won are in `models/model_metadata.json`. On five seeds SMOTE showed no gain
that holds over no resampling (-0.004 to +0.013 of nested AUC), so this card does not claim it helps.

## How well it does

| Measure | Value | From |
|---|---|---|
| AUC, nested cross-validation | 0.888 (folds from 0.809 to 0.928) | 15 outer folds on the 242 |
| AUC, test | 0.957 (95% interval 0.899 to 0.995) | the 61, scored once |
| Recall at the served threshold | 0.92 (0.85 to 0.96): 102 of 111 | out-of-fold, the 242 |
| Precision at the served threshold | 0.63 (0.55 to 0.70): 102 of 162 flagged | out-of-fold, the 242 |
| Test, at the served threshold | 27 of 28 caught, 13 false alarms among 33 without disease | the 61, scored once |
| Brier score | 0.143 out-of-fold, 0.092 on the 61 | |

The number to quote is 0.888, from the project's seed. Over five training seeds it runs from 0.874 to
0.888. The test set gives 0.957, and its interval runs from 0.90 to 0.995: with 61
patients it cannot say more than that. The 61 are an easier draw than the average fold.

The served threshold is 0.222. It comes from counting one missed patient as 3 false alarms, which the
owner decided on November 26, 2026 (`DECISIONS.md`, decision 1). At that threshold two of every three
patients are flagged. The place of the threshold is not a precise number: when the 242 patients are
resampled, the lowest-cost cut at 3 to 1 runs from 0.12 to 0.59. Recall at the served cut runs from 0.87
to 0.97 over the same resamples.

## By sex

Out-of-fold, at the served threshold:

| | Patients | With disease | Caught | False alarms among those without disease | AUC |
|---|---|---|---|---|---|
| Women | 77 | 18 | 15 | 21 of 59 | 0.843 |
| Men | 165 | 93 | 87 | 39 of 72 | 0.850 |

Only 18 women with disease are in the training data. The interval on 15 of 18 runs from 0.59 to 0.96.
This file cannot show that the model works equally well for women.

## What carries the model

Share of the mean absolute SHAP value on the training patients: thallium test result (thal) 23%, vessels
colored (ca) 21%, chest pain type (cp) 18%. Those three hold 61%.

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
   under 0.85, the low end of the interval measured here.
3. An input gains a code the data dictionary does not hold, or a unit changes.
4. A year has passed since the last training.

Retraining means: run `pipeline/00` to `05`, in order, on the new data. Run the tests. Compare the new
nested AUC with 0.888: a change inside the fold spread (0.03) is not a change. Train on the same
operating system each time: Windows and Linux give different trees from the same seed. Make decision 1 again on
the new table. Give the model a new version name, so the old file stays in the repository and going back
is one line in the metadata.

Do not retrain when the drift check shows "watch" once, or when one week's alert is not repeated.
