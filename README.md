# Patient Risk Stratification Pipeline

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange.svg)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.51-brightgreen.svg)](https://shap.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED.svg)](https://docker.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

End-to-end machine learning pipeline that predicts patient heart disease risk from clinical features. Built to demonstrate the full ML workflow: EDA, feature engineering, model selection, explainability, calibration, and containerized deployment. **Phase 2 (May 2026)** added causal inference and survival analysis modules on the NHEFS cohort, extending the project from predictive to explanatory inference.

---

## Headline Results

### Predictive Modeling (UCI Heart Disease)

| Metric | Value |
|---|---|
| Test ROC-AUC (tuned XGBoost) | 0.95 |
| 5-fold CV ROC-AUC | 0.89 |
| Recall at clinical threshold | 0.93 |
| Precision at clinical threshold | 0.81 |
| Brier score (calibration) | 0.092 |
| Test set size | 61 patients |

Five model families compared (Logistic Regression, Random Forest, XGBoost, SVM, PyTorch NN). Tuned XGBoost selected for production due to favorable trade-offs in serving infrastructure, SHAP integration, and scaling behavior. Top three predictive features (`thal`, `ca`, `cp`) accounted for ~75% of feature importance and matched independent EDA correlations.

### Phase 2 Additions (NHEFS, May 2026)

| Module | Estimand | Headline |
|---|---|---|
| Causal Inference — PSM | ATT of cessation on weight change | **+3.27 kg** (95% CI 2.11, 4.43) |
| Causal Inference — G-computation | Independent cross-check | **+3.47 kg** (95% CI 2.58, 4.44) |
| Causal Inference — Refutation | 3 sensitivity tests | All passed |
| Survival — Cox PH | Adjusted HR for cessation on 10-yr mortality | **1.03** (95% CI 0.81, 1.32, p = 0.78) |
| Survival — Cox PH vs RSF | Test-set concordance | **0.802 / 0.804** |
| Survival — PH assumption | Schoenfeld residuals | All 10 covariates passed |

The naive PSM comparison estimated +2.54 kg for the causal effect; after backdoor adjustment, the estimate landed within 0.1 kg of Hernán-Robins' canonical IPW result. The naive Kaplan-Meier comparison showed quitters with significantly worse survival (log-rank p = 0.005), entirely explained by age confounding after Cox PH adjustment.

---

## Tech Stack

**ML & Data:** Python 3.13, pandas, numpy, scikit-learn, XGBoost, PyTorch, imbalanced-learn (SMOTE), SHAP, MLflow

**Causal Inference (Phase 2):** DoWhy, statsmodels, causaldata (NHEFS), NetworkX

**Survival Analysis (Phase 2):** lifelines, scikit-survival

**Serving:** FastAPI, Uvicorn, Pydantic

**Deployment:** Docker (multi-stage), uvicorn ASGI server

**Visualization:** matplotlib, seaborn

---

## Project Structure

---
patient-risk-stratification/
├── data/
│   ├── raw/                              UCI Heart Disease (Cleveland subset)
│   └── processed/                        Train/test splits, post-SMOTE
├── notebooks/
│   ├── 01_eda.ipynb                      Exploratory data analysis
│   ├── 02_modeling.ipynb                 5 models trained, top 3 tuned
│   ├── 03_shap_calibration.ipynb         SHAP + calibration analysis
│   ├── 04_causal_inference.ipynb         PSM + DoWhy + G-computation on NHEFS
│   └── 05_survival_analysis.ipynb        KM + Cox PH + RSF on NHEFS
├── src/
│   ├── app.py                            FastAPI service
│   └── schemas.py                        Pydantic request/response models
├── models/                               Trained XGBoost + SHAP explainer
├── reports/figures/                      Plots (SHAP, calibration, comparison, KM, love plot)
├── Dockerfile                            Multi-stage production image
├── requirements.txt                      Pinned dependencies
└── README.md

---

## Methodology

### 1. Exploratory Data Analysis
Dataset: UCI Heart Disease (Cleveland), 303 patients, 13 clinical features (plus binary target). Target reframed from 5-class severity to binary (disease vs. healthy) for clinical actionability. Distribution: 54% healthy / 46% disease. Demographic skew flagged (68% male) as a fairness concern requiring separate validation by sex before clinical use.

### 2. Preprocessing
Two parallel preprocessing tracks built to match each model family:
- **Tree track** (Random Forest, XGBoost): raw features, no scaling, no one-hot encoding
- **Linear track** (Logistic Regression, SVM, NN): one-hot categoricals (with `drop_first` to avoid the dummy variable trap), StandardScaler on numerical features

All preprocessing fit on training data only. Train/test split (80/20) stratified to preserve class balance. SMOTE applied to training data only, after the split, to prevent leakage. Median imputation for missing `ca` (1.32%) and `thal` (0.66%).

### 3. Modeling
Five model families trained and logged to MLflow with hyperparameters, metrics, and artifacts:
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- SVM (RBF kernel)
- PyTorch feedforward NN (3 layers, dropout, ReLU)

Top three (RF, SVM, XGBoost) tuned via RandomizedSearchCV with 5-fold cross-validation. Final model selected for production based on a combination of CV score (honest generalization) and serving advantages (TreeExplainer for SHAP, scaling behavior, ecosystem).

![Model Comparison](reports/figures/model_comparison.png)

### 4. Explainability: SHAP

SHAP TreeExplainer applied to the tuned XGBoost. Global feature importance and per-patient force plots generated.

![SHAP Beeswarm](reports/figures/shap_beeswarm.png)

The model independently rediscovered a counterintuitive clinical finding from EDA: patients reporting *asymptomatic* chest pain (cp=4) had the highest disease rates. This alignment between EDA and SHAP feature importance gave confidence the model was learning real clinical patterns rather than spurious noise.

Per-patient explanations available via the `/explain` API endpoint, enabling clinicians to verify any prediction:

![SHAP Waterfall: Disease patient](reports/figures/shap_waterfall_disease.png)

### 5. Calibration & Threshold Tuning

Brier score evaluated to assess probability quality. The raw model achieved a Brier score of 0.092, already strong relative to the 0.25 baseline of a "predict 0.5 always" model. Platt scaling via 5-fold CV was tested but degraded Brier to 0.104, likely due to a combination of (a) the model already being near-calibrated, (b) CV variance on the small training set exceeding the calibration gain, and (c) non-monotonic deviation in the reliability curve that a single sigmoid cannot fully correct. The raw model is shipped with documented Brier and reliability curves; isotonic calibration on a held-out calibration set would be the next step with more data.

![Reliability Curve](reports/figures/calibration_comparison.png)

Three operating thresholds were evaluated:
- **Default 0.500**: precision 0.84, recall 0.93
- **F1-optimal 0.533**: precision 0.87, recall 0.93
- **Clinical 0.423** (recall-priority, precision ≥ 0.80): precision 0.81, recall 0.93

The clinical threshold maximizes recall under a precision-≥-0.80 constraint, appropriate for medical screening where false negatives (missed diagnoses) cost more than false positives (follow-up tests).

![Threshold Analysis](reports/figures/threshold_analysis.png)

### 6. Deployment

FastAPI service exposes three endpoints:
- `GET /health` : service liveness
- `POST /predict` : patient features → disease probability + risk tier + clinical threshold applied
- `POST /explain` : patient features → prediction + per-feature SHAP contributions

Pydantic validates inputs against clinical ranges (age 18-100, BP 50-250, etc.) before reaching the model. OpenAPI/Swagger docs auto-generated at `/docs`.

The service is containerized via a multi-stage Dockerfile using a slim Python base, non-root user (`apiuser`), HEALTHCHECK directive, and `--chown` during multi-stage COPY to handle the multi-stage + non-root user permissions clash.

### 7. Causal Inference: ATT of Smoking Cessation on Weight Change

Phase 2 addition focused on a methodologically distinct question: estimating *causal* effects rather than predictive ones. Built on the NHEFS dataset (National Health Epidemiologic Follow-up Study, 1971–1982, 1,629 subjects), the canonical teaching dataset from Hernán & Robins' *Causal Inference: What If*. The analysis follows the DoWhy 4-step framework.

**1. Model — Causal DAG.** Specified the directed acyclic graph for the qsmk → wt82_71 relationship, with nine confounders (age, sex, race, education, smokeintensity, smokeyrs, exercise, active, wt71) acting as common causes of both treatment and outcome.

**2. Identify — Backdoor adjustment.** Applied Pearl's backdoor criterion via DoWhy's automatic identification. The backdoor set blocks every confounding path; IV and frontdoor strategies were not applicable to the DAG.

**3. Estimate — Two methodologically independent estimators:**

| Estimator | Method | ATT | 95% CI |
|---|---|---|---|
| Propensity Score Matching | Hernán-Robins specification (quadratic confounder terms), 1:1 NN on logit-PS, 0.2-SD caliper, without replacement | +3.27 kg | [2.11, 4.43] (bootstrap) |
| G-computation | OLS outcome model with same confounders + quadratic terms, counterfactual prediction | +3.47 kg | [2.58, 4.44] (bootstrap) |
| Reference: Hernán-Robins IPW | Inverse Probability Weighting on full sample | ~+3.4 to +3.5 kg | — |

Three-way convergence within 0.2 kg across methodologically distinct estimators is the primary validity check.

**4. Refute — Three sensitivity tests, all passed:**
- *Placebo treatment:* effect dropped from +2.04 to +0.06 (p = 0.94)
- *Random common cause:* effect unchanged at +2.04 (p = 1.00)
- *Data subset (80%):* effect +2.24, within 10% of original (p = 0.74)

**Diagnostics.** Propensity overlap: both groups span [0.05, 0.78] with no positivity violations. Love plot: 7/9 confounders had |SMD| > 0.10 before matching, 0/9 after; max |SMD| dropped from 0.282 (age) to 0.074. Match retention: 96.3% of treated subjects matched within caliper.

**Interpretation.** The naive comparison suggested smoking cessation caused +2.54 kg of weight gain. After adjusting for confounding via two independent methods, the causal ATT is estimated at +3.27 to +3.47 kg, with bootstrap 95% confidence intervals that do not cross zero. The naive estimate was biased *downward* by older, lighter-smoking, more-educated subjects being over-represented among quitters — characteristics that independently predict less post-cessation weight gain.

### 8. Survival Analysis: 10-Year Mortality and Smoking Cessation

Same NHEFS cohort, time-to-event framing. Mortality follow-up from January 1983 through December 1992; 318 deaths over 1,629 subjects (19.5% event rate).

**Censoring construction.** Built (time, event) columns from raw yrdth/modth/dadth fields per Hernán-Robins Chapter 17 convention. Subjects alive at Dec 1992 right-censored at 120 months.

**Kaplan-Meier estimator.** Non-parametric survival function S(t) overall and stratified by smoking cessation status. Five-year survival 90.4%; ten-year survival 80.5%; median survival not estimable (>50% alive at end of follow-up).

**Log-rank test.** Quitters (n=428, deaths=102, ten-year survival 76.2%) vs non-quitters (n=1201, deaths=216, ten-year survival 82.1%): χ² = 7.73, p = 0.0054 — quitters significantly *worse* in unadjusted analysis. *This is a confounding signal, not a causal effect, as established by the adjusted analysis below.*

**Cox proportional hazards model.** Fit semi-parametric model adjusting for age, sex, race, education, smokeintensity, smokeyrs, exercise, active, and wt71. Adjusted hazard ratio for cessation: **1.035 (95% CI 0.81 to 1.32, p = 0.78)** — fully consistent with no effect after adjustment. Age was the dominant predictor (HR 1.082 per year, p < 0.0005). The naive log-rank finding was entirely explained by confounding.

**Proportional hazards assumption.** Validated via Schoenfeld residuals; all 10 covariates passed at the 0.05 threshold, with no stratification or time-varying coefficients required.

**Random Survival Forest as comparison.** Trained on a 75/25 split with identical covariates. Test-set concordance:

| Model | Test Concordance | Top Predictor |
|---|---|---|
| Cox PH (semi-parametric, PH assumption) | 0.802 | age (HR 1.082) |
| Random Survival Forest (non-parametric, no PH) | 0.804 | age (importance 0.092) |

Two methodologically independent models — one parametric under PH, one non-parametric without — agreed on both predictive performance (~0.80) and covariate ranking. qsmk's RSF permutation importance was 0.0005, confirming the Cox finding of no detectable effect.

**Interpretation.** Smoking cessation showed no detectable effect on 10-year mortality in this observational cohort after adjustment. The unadjusted appearance of worse survival among quitters was a confounding artifact (quitters were 3.4 years older with longer smoking histories). The absence of a protective effect at 10 years does not contradict the well-established RCT evidence on long-term cessation benefits; the follow-up window is shorter than the time horizon over which cessation benefits manifest, and the analysis also faces *selection by indication* — quitting is partially driven by emerging health problems that no measured covariate fully captures.

---

## Reproducing the Results

### Local (Python venv)

```bash
git clone https://github.com/Shrikant-Sharma/patient-risk-stratification.git
cd patient-risk-stratification
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # Mac/Linux
pip install -r requirements.txt
```

Run the notebooks in order: `01_eda.ipynb` → `02_modeling.ipynb` → `03_shap_calibration.ipynb` → `04_causal_inference.ipynb` → `05_survival_analysis.ipynb`.

NHEFS data for notebooks 04 and 05 is loaded directly from the `causaldata` Python package; no manual download required.

Launch the API:
```bash
uvicorn src.app:app --reload --port 8000
```

Then visit http://localhost:8000/docs.

### Docker

```bash
docker build -t patient-risk-api:1.0 .
docker run -d --name patient-risk-api -p 8000:8000 patient-risk-api:1.0
```

Then visit http://localhost:8000/docs.

---

## Lessons Learned

A few engineering and methodological issues debugged during build, included here because the diagnosis matters as much as the result:

**1. Feature-order training/serving skew.** XGBoost's JSON serialization stores feature names and validates order at inference. The training pipeline ordered numerical features before categoricals; the API initially sent them in natural data dictionary order, causing a 500 error on first POST. Fixed by aligning column order in the API. Production lesson: save column order alongside the model in metadata; don't trust hardcoded ordering.

**2. Multi-stage Docker + non-root user permissions clash.** Builder-stage dependencies came over with root ownership and 700 permissions; the non-root runtime user (`apiuser`) couldn't execute uvicorn. Fixed with `COPY --from=builder --chown=apiuser:apiuser`. Production lesson: when combining multi-stage builds with non-root users (both security best practices), ownership must be transferred explicitly during COPY.

**3. Calibration is a hypothesis, not a tool.** Platt scaling degraded Brier on this dataset despite intuition suggesting it would help. Three reasons: model was already near-calibrated, CV calibration variance on 262 training rows exceeded the calibration gain, and the reliability curve had non-monotonic deviation that a single sigmoid couldn't fully correct. Reported both metrics and shipped the raw model with documented reliability.

**4. Two independent estimators converging is the strongest evidence pattern (Phase 2).** For both causal inference (PSM + G-computation) and survival analysis (Cox PH + RSF), I deliberately fit two methodologically distinct estimators on the same problem. Three-way convergence within 0.2 kg on the causal effect, and within 0.002 on survival concordance — across models with completely different assumptions — is far more credible than any single point estimate. Production lesson: when defensibility matters more than speed, fit a method-independent cross-check.

**5. Library defaults can systematically bias causal estimates (Phase 2).** DoWhy's default `propensity_score_matching` produced +2.04 kg versus my custom Hernán-Robins implementation's +3.27 kg — a 1.2 kg gap traceable to four specific defaults: L2 penalty (vs unpenalized), linear-only confounder terms (vs quadratics on continuous variables), no caliper (vs 0.2-SD), and matching with replacement (vs without). Production lesson: causal libraries are most valuable for the framework and refutation tests; for the actual estimate, specify the propensity model and matching parameters explicitly rather than accepting defaults.

**6. Unadjusted Kaplan-Meier curves can actively reverse a true causal effect (Phase 2).** On NHEFS 10-year mortality, the unadjusted log-rank showed quitters with significantly worse survival (p = 0.005). After Cox PH adjustment for age and other confounders, the effect attenuated to HR 1.03 (p = 0.78) — fully null. The unadjusted picture wasn't just imprecise, it was directionally wrong. Production lesson: never report unadjusted survival comparisons as evidence of treatment effect in observational data without an adjusted analysis alongside.

---

## Limitations

### Predictive Modeling (Phase 1)

- **Test set size (n=61) is small.** AUC has ~0.02 standard error; differences <2% are statistical noise. CV scores are reported alongside test scores as the more honest generalization estimate.
- **Demographic skew (68% male).** Model performance should be validated separately on female patients before any clinical deployment.
- **Single-source dataset.** UCI Heart Disease (Cleveland subset) is well-studied and clean. Real clinical data has substantially more missingness, more heterogeneous quality, and feature drift over time. Production deployment would require ongoing model monitoring and retraining triggers.

### Causal & Survival (Phase 2, NHEFS)

- **Unmeasured confounding is untestable.** The unconfoundedness assumption is fundamental to both PSM and G-computation, and cannot be empirically verified. Sensitivity analyses (e-values, Rosenbaum bounds) are a natural extension.
- **No treatment-effect heterogeneity modeling.** ATT here is an average; effect modification by age, sex, or smoking intensity is plausible and would motivate causal forests or BART-based extensions.
- **10-year follow-up window is shorter than the multi-decade horizon** over which smoking cessation's mortality benefits manifest. The null Cox PH result for cessation does not contradict long-term RCT evidence of cessation benefits.
- **Selection by indication.** In observational data, quitting is partially driven by emerging health problems that no measured covariate fully captures. This is the standard limitation of observational causal inference on smoking cessation; RCTs (Lung Health Study) are the source of definitive causal evidence.
- **Reference for full methodology:** Hernán MA, Robins JM. *Causal Inference: What If.* Chapters 12 and 17. Boca Raton: Chapman & Hall/CRC, 2020.

---

## Future Work

- **AIPW (Augmented Inverse Probability Weighting)** for doubly-robust causal estimation — consistent if either the propensity or outcome model is correctly specified
- **Causal Forests** (econml, on Python 3.12 venv) for heterogeneous treatment effect estimation across subgroups
- **AWS Lambda + API Gateway** public deployment of the FastAPI service, behind a public Swagger UI
- **Agentic explanation layer** with LangGraph for natural-language clinical summaries
- **Larger longitudinal dataset integration** (NHANES, MIMIC subsets) to enable deeper validation and richer covariate sets
- **Model monitoring** with drift detection and automated retraining triggers

---

## License

MIT.See [LICENSE](LICENSE).

---

## Contact

[Shrikant Sharma](https://www.linkedin.com/in/shrikant-sharma) · GitHub: [@Shrikant-Sharma](https://github.com/Shrikant-Sharma)