# Run summary

My own run of version 2, beside the reference numbers that shipped with it. Every row is read from the result files of both runs and marked same or different.

| | |
|---|---|
| My run | November 26, 2026, 11:05 to 11:09 (UTC-05:00). 9 of 9 steps passed in 220 seconds |
| My machine | Windows-11-10.0.26320-SP0, AMD Ryzen 7 4800H with Radeon Graphics, 16 logical cores |
| My versions | Python 3.13.13, numpy 2.4.4, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.8.0, imbalanced-learn 0.14.1, xgboost 3.2.0, shap 0.51.0, numba 0.68.1, causaldata 0.1.5, fastapi 0.136.1, uvicorn 0.54.0, pydantic 2.13.3, httpx 0.28.1, pytest 9.2.0 |
| Reference | The build of November 23, 2026: Linux, 2 CPU cores, Python 3.13, the same pinned packages |
| Rows below | 44 of 96 are the same |
| Every stored value | 788 of 1120 values in the seven JSON files are the same. 3 describe the run itself and differ as expected. 329 more differ |

## Why my numbers differ from the reference

One cause explains every difference. XGBoost samples rows and columns for each tree. With the same seed it draws a different sample on Windows than on Linux. This project's search always samples (`subsample` and `colsample_bytree` are drawn between 0.6 and 1), so every fit grows different trees on the two systems. The reference was built on Linux. My run is on Windows.

Checked on November 26, 2026. The Linux side ran on a Linux machine with the same pinned packages, on my result files.

| Check | Result |
|---|---|
| Numbers that do not pass through an XGBoost fit: the data, the split, the causal checks, the drift limits of the 13 inputs | The same, to the last stored digit |
| The 40 candidate settings of the random search | The same list on both systems. A different candidate wins: draw 7 in my run, draw 15 in the reference |
| Tree 1 of my served model, beside the same settings trained on Linux | Row sampling kept 219 of 262 rows on Windows and 218 on Linux |
| My settings trained on Linux, against my own scores for the 61 held-back patients | Different: mean gap 0.033, largest gap 0.198 |
| The same training twice on one system | The same scores. On Windows, step 03 trained the model again and reproduced step 01's 61 scores. On Linux, two fits agreed exactly |
| My model file, trained on Windows, scored on Linux | My recorded scores, to 7 decimals. Training differs between the systems. Scoring does not |
| The 29 tests and the live check on Linux, with my model files | Passed |

What follows from it:

- My numbers are the ones this repository reports. They replace the reference numbers in the README, the model card and the decisions file.
- The two runs are two draws of one recipe, and they agree on every conclusion. SMOTE inside the folds is chosen. The raw score is served. The nested AUC is 0.888 here and 0.886 in the reference.
- The threshold moves most. At 3 false alarms for a miss the cut is 0.222 here and 0.136 in the reference. With 242 patients, where the cut lands hangs on a few patients' scores.
- The pinned packages reproduce these files on the same operating system. They do not reproduce them across Windows and Linux.

## Steps and time

| Step | Seconds | Result |
|---|---|---|
| 00 | 4.4 | passed |
| 01 | 92.5 | passed |
| 02 | 3.6 | passed |
| 03 | 25.5 | passed |
| 04 | 43.5 | passed |
| 05a | 9.9 | passed |
| 05b | 18.2 | passed |
| 06 | 9.1 | passed |
| tests | 13.7 | passed |

## The three scores (step 01)

| Number | My run | Reference | |
|---|---|---|---|
| Patients, training, held back | 303, 242, 61 | 303, 242, 61 | same |
| With disease: all, training, held back | 139, 111, 28 | 139, 111, 28 | same |
| A. The notebooks' way: rows before and after SMOTE | 242, 262 | 242, 262 | same |
| A. The notebooks' folds: cross-validated AUC | 0.887 | 0.890 | **different** |
| A. The notebooks' folds: test AUC | 0.950 | 0.960 | **different** |
| A. Shuffled folds: cross-validated AUC (the leak, like for like) | 0.918 | 0.917 | **different** |
| A. Shuffled folds: test AUC | 0.947 | 0.950 | **different** |
| B. SMOTE inside the folds: best draw's own AUC | 0.899 | 0.896 | **different** |
| B. Nested AUC, mean of 15 folds (the number to quote) | 0.888 | 0.886 | **different** |
| B. Nested AUC: spread, lowest fold, highest fold | 0.031, 0.809, 0.928 | 0.026, 0.815, 0.927 | **different** |
| B. Nested AUC by repeat | 0.870, 0.891, 0.889 | 0.874, 0.884, 0.889 | **different** |
| B. Test AUC on the 61 | 0.957 | 0.946 | **different** |
| B. Test AUC, 95% interval | 0.899, 0.995 | 0.884, 0.990 | **different** |
| B. The eight settings the search chose | 369, 4, 0.1449, 0.8282, 0.6125, 4.211, 0.7273, 0.3265 | 185, 5, 0.1897, 0.7183, 0.6812, 4.714, 0.8805, 0.6244 | **different** |
| C. No resampling: best draw's own AUC | 0.899 | 0.895 | **different** |
| C. Nested AUC, mean of 15 folds | 0.875 | 0.877 | **different** |
| C. Test AUC on the 61 | 0.949 | 0.936 | **different** |
| Standard error used by the rule | 0.0081 | 0.0077 | **different** |
| Recipe chosen by the rule | B_smote_inside_folds | B_smote_inside_folds | same |

## Calibration and threshold (step 02)

| Number | My run | Reference | |
|---|---|---|---|
| Brier score out-of-fold: raw, Platt, isotonic | 0.1427, 0.1417, 0.1506 | 0.1406, 0.1403, 0.1372 | **different** |
| Gap between predicted and observed: raw, Platt, isotonic | 0.049, 0.069, 0.049 | 0.040, 0.056, 0.030 | **different** |
| Platt minus raw, 95% interval | -0.0045, +0.0026 | -0.0028, +0.0022 | **different** |
| Isotonic minus raw, 95% interval | -0.0026, +0.0183 | -0.0141, +0.0077 | **different** |
| Calibration chosen | raw | raw | same |
| A miss costs 1: threshold, flagged, caught, missed, false alarms | 0.603, 93, 80, 31, 13 | 0.634, 93, 81, 30, 12 | **different** |
| A miss costs 2: threshold, flagged, caught, missed, false alarms | 0.537, 107, 86, 25, 21 | 0.374, 130, 94, 17, 36 | **different** |
| A miss costs 3: threshold, flagged, caught, missed, false alarms | 0.222, 162, 102, 9, 60 | 0.136, 181, 107, 4, 74 | **different** |
| A miss costs 5: threshold, flagged, caught, missed, false alarms | 0.118, 197, 109, 2, 88 | 0.136, 181, 107, 4, 74 | **different** |
| A miss costs 10: threshold, flagged, caught, missed, false alarms | 0.113, 197, 109, 2, 88 | 0.133, 181, 107, 4, 74 | **different** |
| Cost ratio used (the placeholder until decision 1) | 3 | 3 | same |
| Served threshold | 0.2215 | 0.1364 | **different** |
| Out-of-fold at the served threshold: caught, missed, false alarms, flagged | 102, 9, 60, 162 | 107, 4, 74, 181 | **different** |
| Recall out-of-fold, with its exact interval | 0.919, 0.852, 0.962 | 0.964, 0.910, 0.990 | **different** |
| Precision out-of-fold, with its exact interval | 0.630, 0.550, 0.704 | 0.591, 0.516, 0.664 | **different** |
| The 61, once: caught, missed, false alarms, cleared | 27, 1, 13, 20 | 28, 0, 19, 14 | **different** |
| The 61, once: lowest true recall the count allows | 0.817 | 0.877 | **different** |
| Brier score on the 61, raw | 0.0922 | 0.1000 | **different** |
| Version 1's cut (0.4226) out-of-fold: caught, recall, precision | 92, 0.829, 0.724 | 90, 0.811, 0.726 | **different** |
| A cut of 0.5 out-of-fold: caught, missed, false alarms | 87, 24, 26 | 87, 24, 29 | **different** |

## The served model (step 03)

| Number | My run | Reference | |
|---|---|---|---|
| Model version, threshold, cost ratio | xgb_v2, 0.2215, 3 | xgb_v2, 0.1364, 3 | **different** |
| Calibration served | raw | raw | same |
| Upper cut of the tiers (MODERATE to HIGH) | 0.6033 | 0.6344 | **different** |
| Women, out-of-fold: patients, with disease, caught, false alarms, without disease | 77, 18, 15, 21, 59 | 77, 18, 17, 29, 59 | **different** |
| Women, out-of-fold: AUC | 0.843 | 0.842 | **different** |
| Men, out-of-fold: patients, with disease, caught, false alarms, without disease | 165, 93, 87, 39, 72 | 165, 93, 90, 45, 72 | **different** |
| Men, out-of-fold: AUC | 0.850 | 0.856 | **different** |
| Share of mean absolute SHAP: thal, ca, cp | 0.227, 0.207, 0.180 | 0.245, 0.202, 0.184 | **different** |
| Chance of a false drift alert at 30, 50, 100 requests | 2.9%, 2.9%, 2.8% | 2.8%, 2.9%, 2.9% | **different** |
| Chance of a watch at 30, 50, 100 requests | 9.6%, 8.5%, 9.4% | 9.7%, 8.5%, 9.4% | **different** |
| The 13 inputs: measured limits at 50 requests | 13 limits, from 0.128 to 0.271 | 13 limits, from 0.128 to 0.271 | same |
| The 13 inputs: strict limits at 50 requests | 13 limits, from 0.198 to 0.387 | 13 limits, from 0.198 to 0.387 | same |
| The 13 inputs: limits at 30 and 100 requests (52 values) | from 0.067 to 0.682 | from 0.067 to 0.682 | same |
| The score: measured and strict limit at 50 requests | 0.250, 0.338 | 0.263, 0.400 | **different** |
| Data file: fingerprint (first 12 characters) and patients | a74b7efa387b, 303 | a74b7efa387b, 303 | same |
| Built on | 2026-11-26 | 2026-11-23 | different, as expected: the date of the run |
| Versions: Python | 3.13.13 | 3.13.17 | different, as expected: the Python patch version |
| Versions: xgboost, scikit-learn, imbalanced-learn, numpy | 3.2.0, 1.8.0, 0.14.1, 2.4.4 | 3.2.0, 1.8.0, 0.14.1, 2.4.4 | same |

## The causal checks (step 04)

| Number | My run | Reference | |
|---|---|---|---|
| People, quitters, matched pairs | 1566, 403, 388 | 1566, 403, 388 | same |
| Unadjusted difference, kg | 2.541 | 2.541 | same |
| Headline estimate, kg | 3.270 | 3.270 | same |
| Largest standardized gap before and after matching | 0.282, 0.075 | 0.282, 0.075 | same |
| Interval from the pairs alone, kg | +2.11, +4.43 | +2.11, +4.43 | same |
| Placebo: mean, low, high, kg | -0.01, -0.98, +0.96 | -0.01, -0.98, +0.96 | same |
| Placebo: share of runs as large as the headline | 0% | 0% | same |
| Random common cause: mean, low, high, kg | +3.43, +2.96, +3.94 | +3.43, +2.96, +3.94 | same |
| A random 80%: mean, low, high, kg | +3.35, +2.56, +4.16 | +3.35, +2.56, +4.16 | same |
| Caliper 0.1, 0.2, 0.3: estimate, kg | +3.35, +3.27, +3.33 | +3.35, +3.27, +3.33 | same |
| Caliper 0.1, 0.2, 0.3: pairs | 387, 388, 389 | 387, 388, 389 | same |
| Whole procedure resampled: mean, low, high, kg | +3.38, +2.24, +4.53 | +3.38, +2.24, +4.53 | same |

## The drift replay (step 05a)

| Number | My run | Reference | |
|---|---|---|---|
| As published: result | ok | ok | same |
| As published: inputs over their limit | none | none | same |
| As published: inputs over their strict limit | none | none | same |
| As published: score over its limit, share flagged | no, 68% | no, 78% | **different** |
| As published: caught, with disease, false alarms | 27, 28, 13 | 28, 28, 19 | **different** |
| As published: of 200 draws, alert, watch, ok | 7, 45, 148 | 7, 45, 148 | same |
| Older clinic: result | alert | alert | same |
| Older clinic: inputs over their limit | age, ca | age, ca | same |
| Older clinic: inputs over their strict limit | age | age | same |
| Older clinic: score over its limit, share flagged | no, 80% | no, 92% | **different** |
| Older clinic: of 200 draws, alert, watch, ok | 200, 0, 0 | 200, 0, 0 | same |
| Cuff reads high: result | alert | alert | same |
| Cuff reads high: inputs over their limit | trestbps | trestbps | same |
| Cuff reads high: inputs over their strict limit | trestbps | trestbps | same |
| Cuff reads high: score over its limit, share flagged | no, 68% | no, 78% | **different** |
| Cuff reads high: caught, with disease, false alarms | 27, 28, 14 | 28, 28, 19 | **different** |
| Cuff reads high: of 200 draws, alert, watch, ok | 200, 0, 0 | 200, 0, 0 | same |
| No chest pain: result | alert | alert | same |
| No chest pain: inputs over their limit | cp, exang, ca, thal | cp, exang, ca, thal | same |
| No chest pain: inputs over their strict limit | cp, exang | cp, exang | same |
| No chest pain: score over its limit, share flagged | yes, 92% | yes, 98% | **different** |
| No chest pain: of 200 draws, alert, watch, ok | 200, 0, 0 | 200, 0, 0 | same |

## The drift limits: share of 500 draws of 50 with an alert (step 05b)

| Number | My run | Reference | |
|---|---|---|---|
| Training patients, nothing changed: fixed 0.10, fixed 0.25, measured, any input, as built | 75.6%, 2.6%, 3.8%, 0.6% | 74.0%, 2.6%, 3.6%, 0.4% | **different** |
| As published: fixed 0.10, fixed 0.25, measured, any input, as built | 100.0%, 18.0%, 26.8%, 4.2% | 100.0%, 18.0%, 26.8%, 4.2% | same |
| Older clinic: fixed 0.10, fixed 0.25, measured, any input, as built | 100.0%, 100.0%, 100.0%, 100.0% | 100.0%, 100.0%, 100.0%, 100.0% | same |
| Cuff reads high: fixed 0.10, fixed 0.25, measured, any input, as built | 100.0%, 100.0%, 100.0%, 100.0% | 100.0%, 100.0%, 100.0%, 100.0% | same |
| No chest pain: fixed 0.10, fixed 0.25, measured, any input, as built | 100.0%, 100.0%, 100.0%, 100.0% | 100.0%, 100.0%, 100.0%, 100.0% | same |

## The files

| File | Against the reference |
|---|---|
| `results/evaluation.json` | **different**: 55 of 82 stored values |
| `results/threshold_calibration.json` | **different**: 152 of 179 stored values |
| `results/causal_checks.json` | identical apart from Windows line endings |
| `results/drift_replay.json` | **different**: 16 of 167 stored values |
| `results/drift_limits.json` | **different**: 3 of 26 stored values |
| `models/model_metadata.json` | **different**: 86 of 349 stored values |
| `models/known_answers.json` | **different**: 20 of 280 stored values |
| `results/out_of_fold_predictions.csv` | **different**: largest gap between two scores 2.39e-01 |
| `results/test_predictions.csv` | **different**: largest gap between two scores 2.28e-01 |
| `models/xgb_v2.json` | **different** (fingerprints 39168a610e and 9dd5379b07) |

## Every difference

Expected, because they describe the run and not the model:

| File | Value | My run | Reference | Why |
|---|---|---|---|---|
| `results/evaluation.json` | `seconds` | 88 | 132 | how long step 01 took |
| `models/model_metadata.json` | `built` | 2026-11-26 | 2026-11-23 | the date of the run |
| `models/model_metadata.json` | `versions.python` | 3.13.13 | 3.13.17 | the Python patch version |

All other differences, by file. The section near the top says why they differ.

| File | Stored values | Same | Different |
|---|---|---|---|
| `results/evaluation.json` | 82 | 27 | 55 |
| `results/threshold_calibration.json` | 179 | 27 | 152 |
| `results/causal_checks.json` | 37 | 37 | 0 |
| `results/drift_replay.json` | 167 | 151 | 16 |
| `results/drift_limits.json` | 26 | 23 | 3 |
| `models/model_metadata.json` | 349 | 263 | 86 |
| `models/known_answers.json` | 280 | 260 | 20 |

The single values, for files where few differ:

| File | Value | My run | Reference |
|---|---|---|---|
| `results/drift_limits.json` | `share_of_draws_with_an_alert.training patients, nothing changed.as built: score, or two inputs, or one input over its strict limit` | 0.006 | 0.004 |
| `results/drift_limits.json` | `share_of_draws_with_an_alert.training patients, nothing changed.fixed 0.10, any input or the score` | 0.756 | 0.74 |
| `results/drift_limits.json` | `share_of_draws_with_an_alert.training patients, nothing changed.measured limit, any input or the score` | 0.038 | 0.036 |

