# Decisions

Version 2 of this project left two decisions open for me. For each one this file gives the question, the
options, what each option costs and gains, where the evidence is, and then my choice and its reason. One
change of my own follows them. Part 2 lists the choices that were already built in, with their reasons.

Every number here is from my own run of November 26, 2026, on Windows 11 and Python 3.13 with the pinned
packages. The build's reference run of November 23 was on Linux and gave slightly different numbers:
`results/RUN_SUMMARY.md` sets the two side by side and says why.

## Part 1. My two decisions and my own change

### Decision 1. What a miss costs against a false alarm

**The question.** The model gives a probability. A threshold turns it into "flag this patient" or "do
not". Where the threshold sits depends on one judgment: how many patients flagged without disease is one
missed patient with disease worth? That number is in `decision.json`. I set it to 3 on November 26, 2026.

**The options.** Each row is the lowest-cost threshold for that ratio, measured on out-of-fold scores for
the 242 training patients (111 with disease). No row used the 61 test patients.

| A miss costs | Threshold | Flagged | Caught | Missed | False alarms | Recall | Precision |
|---|---|---|---|---|---|---|---|
| 1 false alarm | 0.603 | 93 | 80 | 31 | 13 | 0.72 | 0.86 |
| 2 false alarms | 0.537 | 107 | 86 | 25 | 21 | 0.77 | 0.80 |
| 3 false alarms | 0.222 | 162 | 102 | 9 | 60 | 0.92 | 0.63 |
| 5 false alarms | 0.118 | 197 | 109 | 2 | 88 | 0.98 | 0.55 |
| 10 false alarms | 0.113 | 197 | 109 | 2 | 88 | 0.98 | 0.55 |

These rows are from my run of November 26, 2026, on Windows. The build's reference run, on Linux, gave
other rows (0.136 at 3 to 1), because XGBoost samples rows differently on the two systems. See
`results/RUN_SUMMARY.md`.

**What each choice costs and gains.** At 2 to 1 the model misses 25 of 111 patients with disease and
sends 21 people without it for a follow-up test. At 3 to 1 it misses 9 and sends 60, and two of every
three patients are flagged. At 5 to 1 it misses 2 and sends 88, and four of every five are flagged. One
more patient caught costs 2.4 false alarms between the rows for 2 and 3, and 4 false alarms between the
rows for 3 and 5. The cut of 0.222 is the lowest-cost cut for every ratio from 2.5 to 3.9.

**What to weigh.** What follows a flag here? If it is a cheap test, a high ratio is easy to defend. If it
is an invasive one, it is not. The first version's threshold (0.423) was picked on the 61 test patients,
with a floor of 0.80 on precision, and then reported on the same 61. On the 242 out-of-fold scores of the
rebuilt model the same cut catches 92 of 111 (recall 0.83), not the 0.93 that was reported.

**To change it.** Edit `decision.json`, then run `python pipeline/02_threshold_and_calibration.py` and
`python pipeline/03_train_final.py`. The tests check that the served threshold matches the decision.

**My ratio and the reason (November 26, 2026).** 3 to 1, which gives a threshold of 0.222. I asked for a
recommended choice, checked it against the table, and took it. Three reasons. First, the label in this
data comes from angiography, an invasive test. If a flag means "send for angiography", a false alarm is
not cheap, so I do not go past 3. Second, the choice is stable: 0.222 is the lowest-cost cut for every
ratio from 2.5 to 3.9, so it does not hang on the exact number. Third, the model still sorts patients:
at 3 to 1 it clears one patient in three, and at 5 to 1 it flags four of every five. The cost is stated:
9 of 111 patients with disease are missed out-of-fold (recall 0.92, exact interval 0.85 to 0.96), and 1
of 28 among the held-back patients. If a flag led only to a non-invasive test, I would move to 5 to 1.
The ratio is a clinical judgment, so the table stays in view.

### Decision 2. The drift limits

**The question.** The drift check compares the last 50 requests with the training patients, input by
input, using the population stability index (PSI). When should it alert?

**The options.** `python pipeline/05_drift_monitor.py --limits` tries four rules on 500 draws of 50
patients from each stream. The table gives the share of draws in which the rule alerts.

| Rule | Training patients, nothing changed | 61 test patients, nothing changed | Older clinic | Cuff reads 15 high | No chest pain |
|---|---|---|---|---|---|
| Fixed limit 0.10 on any input or the score | 76% | 100% | 100% | 100% | 100% |
| Fixed limit 0.25 on any input or the score | 3% | 18% | 100% | 100% | 100% |
| Measured limit (99 in 100) on any input or the score | 4% | 27% | 100% | 100% | 100% |
| As built: the score over its limit, or two inputs, or one input over its strict limit (999 in 1,000) | 0.6% | 4% | 100% | 100% | 100% |

**What each choice costs and gains.** The textbook limit of 0.10 was written for samples of thousands. At
50 requests it alerts three times in four when nothing has changed, so nobody would read it. Any rule
that alerts on a single input pays for 14 separate chances to be wrong. The rule as built asks for more
evidence before it alerts and still caught all three real changes in every draw. Its cost is that a small
change in one input shows as "watch" and not as an alert. On the build's own simulation, with nothing
changed, a window gives a false alert 2.9% of the time and a watch 8.5% of the time.

**What to weigh.** The 61 test patients sit above the training patients in every row, though nothing
changed. They are one fixed sample of 61, and by chance their resting ECG codes are spread differently.
That is what a small reference looks like. Also weigh this: a drift alert says the patients look
different. It does not say the model is wrong. In the cuff stream the alert fired, and the model flagged
41 of the 61 patients where it had flagged 40.

**To change it.** The two percentiles and the windows are at the top of `src/monitoring.py`. The rule is
in `drift_report`. Run step 3 again to measure the new limits.

**My rule and the reason (November 26, 2026).** The rule as built: alert when the score is over its
limit, or two inputs are, or one input is over its strict limit. I asked for a recommended choice and
took it. In my run it is the only rule that stays quiet on both unchanged streams (0.6% and 4% of draws)
and still alerts on all three shifted streams in every draw. An alert that fires three times in four
when nothing has changed would not be read. The cost: a small change in one input shows as "watch" and
not as an alert, and the limits rest on 242 training patients.

### My own change: how far the threshold moves

**Where it started.** My Windows run and the Linux reference run are the same recipe. At 3 to 1 one put
the threshold at 0.222 and the other at 0.136. So I asked how far the cut moves when the patients change.

**What I ran.** `python pipeline/07_threshold_bootstrap.py` resamples the 242 out-of-fold patients 2,000
times and finds the lowest-cost cut again each time. It does not use the 61 held-back patients.

**What was expected, written before it ran.** I asked for the prediction to be picked for me, and it was
this: at 3 to 1 the middle 95% of the cuts will run from about 0.12 to 0.54, fewer than half of the
resamples will land near 0.222, and recall at the served cut will stay between about 0.86 and 0.97.

**The result.** Run on November 26, 2026, on my machine. The prediction held, and the cut moves even more
than expected at the top end.

| A miss costs | Cut on all 242 | Middle 95% of 2,000 resampled cuts | Resamples within 0.02 of that cut |
|---|---|---|---|
| 1 false alarm | 0.603 | 0.500 to 0.740 | 29% |
| 2 false alarms | 0.537 | 0.205 to 0.688 | 28% |
| 3 false alarms | 0.222 | 0.118 to 0.588 | 40% |
| 5 false alarms | 0.118 | 0.118 to 0.339 | 57% |
| 10 false alarms | 0.113 | 0.033 to 0.203 | 76% |

At the served cut of 0.222, over the same resamples: recall 0.87 to 0.97, precision 0.55 to 0.71, and
61% to 73% of patients flagged.

**What I take from it.** The place of the cut is the least stable number in this project. With 242
patients, a 3 to 1 ratio can land anywhere from 0.12 to 0.59. What the cut delivers is steadier: recall
stays near 0.9. So I quote the recall with its interval, I do not present 0.222 as a precise number,
and before any real use the threshold would be set again on more patients. Nothing was changed in the
served model because of this. The full output is `results/threshold_bootstrap.json`.

### Two checks at the close (November 27, 2026)

**Weighting beside matching.** I had said weighting would be steadier than matching without measuring it.
`python pipeline/08_weighting_check.py` uses the same propensity model and gives both estimators the same
noise variable, the same 80% of people and the same shuffle in each of 200 runs.

| Check | Matching, middle 95% of runs | Weighting, middle 95% of runs |
|---|---|---|
| A made-up confounder added | +2.92 to +3.90 kg | +3.29 to +3.39 kg |
| A random 80% of people | +2.55 to +4.16 kg | +2.89 to +3.80 kg |
| Placebo: who quit is shuffled | -0.99 to +0.84 kg | -0.89 to +0.84 kg |

On all 1,566 people matching gives +3.27 kg and weighting +3.34 kg. So the claim holds for the noise
check, where weighting's spread is about a thirteenth of matching's. On a subset it is steadier by less,
and on the placebo the two are alike. The headline stays the matched estimate, as the notebooks built it.

**The seed rule.** Two sentences quoted a gain of one trained recipe over another from one seed.
`python pipeline/09_seed_check.py` trains both sides again on five seeds. The split and the outer folds
stay fixed.

| Seed | Nested AUC with SMOTE | Without | SMOTE's gain | The leak |
|---|---|---|---|---|
| 42, the project's | 0.888 | 0.875 | +0.013 | +0.019 |
| 7 | 0.879 | 0.881 | -0.002 | -0.004 |
| 11 | 0.879 | 0.878 | +0.001 | +0.004 |
| 23 | 0.874 | 0.875 | -0.001 | -0.003 |
| 101 | 0.881 | 0.885 | -0.004 | +0.015 |

Both ranges cross zero, so neither is quoted as a gain any more. The nested AUC itself runs from 0.874 to
0.888, and the project's seed is the top of that range, so 0.88 is the fair round number. The served
model was not changed: it was chosen by a rule set before the numbers, on the project's seed.

## Part 2. Decisions already built in

| # | The choice | Why | What it costs |
|---|---|---|---|
| 1 | Imputing and SMOTE sit inside one pipeline with the model | In cross-validation they are then fitted on the training folds only. In version 1 SMOTE ran once before the search, so a synthetic patient could sit in training while the real patient it was built from was scored | Little, as it turned out. SMOTE added only 20 rows to 242, so the leak was too small to measure: over five seeds the leaky search scored between 0.004 lower and 0.019 higher than the clean one. Version 1's reported 0.89 holds up |
| 2 | The score that is quoted comes from nested cross-validation, repeated three times | The search picks the best of 40 draws. The best draw's own score is flattered by the picking. Nesting scores the whole recipe | 15 searches where version 1 ran one, about two minutes |
| 3 | SMOTE stays, by a rule set before the numbers | With SMOTE inside the folds the nested AUC is 0.888, without it 0.875. The rule was: the simpler recipe unless resampling is ahead by more than one standard error | On five seeds SMOTE's gain runs from -0.004 to +0.013, and the rule chose it on one seed of five, the project's own. So I do not claim SMOTE helps. The classes are 111 to 131 in training, so there was little to balance |
| 4 | The threshold and the calibration are chosen on out-of-fold scores | Each of the 242 patients is scored by a model that never saw that patient, so the 61 test patients stay unseen until the end | The out-of-fold models differ a little from the served one |
| 5 | The 61 test patients are scored once, and reported with exact intervals | 27 of 28 caught sounds close to certainty. The exact interval says the true rate could be as low as 0.82 | A less impressive sentence |
| 6 | The raw score is served, with no calibration layer | Neither Platt scaling nor isotonic regression lowered the Brier score by more than chance. The rule was fixed first | None in my run: isotonic was worse (0.151 against 0.143) and Platt was level (0.142). In the Linux reference run isotonic looked a little better and was still not taken |
| 7 | The model is served from XGBoost's own file and one JSON metadata file | A pickle loads only under the library versions that wrote it, and loading one can run code. Version 1 needed a pickled explainer that was not in the repository | The calibration and the drift check had to be written as plain arithmetic |
| 8 | SHAP values come from XGBoost itself (`pred_contribs`) | The same exact tree SHAP values, without the shap library and its dependencies. A test checks them against the shap library | The API image has no shap plots |
| 9 | The feature order is read from the metadata | Version 1 had the order written twice, once in training and once in the API | None |
| 10 | Coded inputs accept only the codes in the data dictionary | Version 1 accepted any integer for thal. A code the model never saw gets a confident answer | A new code needs a new model and a new dictionary entry |
| 11 | The risk tiers are tied to the threshold | In version 1 the tiers cut at 0.30 and 0.70 and the flag at 0.42, so a flagged patient could be "LOW" | The tier names mean something different from version 1's |
| 12 | The request log holds the 13 inputs and the score, and no identifier | It is what the drift check needs and nothing more | A request cannot be traced back to a person, by design |
| 13 | A log that cannot be written never fails a prediction | Monitoring serves the model, not the other way round | A full disk loses log lines silently |
| 14 | The weekly check marks its own requests, and they are not logged | 20 known patients a week would otherwise fill the drift window | One header to remember |
| 15 | Drift limits are measured at the window size, not taken from a textbook | See decision 2 | The limits depend on 242 training patients |
| 16 | The causal checks wrap the whole estimator in one function | A placebo test has to refit the propensity model and match again. Version 1 ran DoWhy's refuters on DoWhy's own estimator (2.04 kg), not on the one that gave the headline 3.27 kg | The DoWhy framing is no longer what the checks run on |
| 17 | Two intervals are shown for the causal estimate | The pairs-only interval treats the matching as fixed. The other resamples people and matches again | Bootstrap intervals for matching are approximate, and the text says so |
| 18 | Render's free plan serves the API | No card and no bill. Tests run on each push and a check runs each week from GitHub Actions | The service sleeps after 15 minutes, took 27 seconds to wake when I measured it, and loses its log on restart |
| 19 | The Lambda function of version 1 stays, and now runs version 2 | Its address is the one I had already shared, so on November 26, 2026 I pointed the same function at version 2's image. No new function, and version 1's image is kept as the way back | One function stays live. It bills for requests only, and a few cents a month for the stored images |
