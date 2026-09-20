# Design Document — HR Employee Attrition Analysis

## 1. Overview

| | |
|---|---|
| **Project** | Employee attrition/retention ML analysis |
| **Repo** | `github/IFACET-iit-kanpur` |
| **Dataset** | IBM HR Analytics Employee Attrition & Performance (fictional, IBM data science team) |
| **Shape** | 1,470 employees × 35 columns, **0 missing values**, target `Attrition` |
| **Class balance** | 237 attrited (16.1%) vs 1,233 retained (83.9%) — imbalanced |
| **Goal** | Identify factors driving attrition and deliver (a) a predictive model, (b) explainable business rules |

## 2. Why this document exists

The repository contained **six overlapping Jupyter notebooks** that re-run the
same analysis with inconsistent code, hard-coded Colab paths
(`/content/...`), duplicated cells, broken imports and several methodology
errors (detailed in §5). This consolidation rewrites the analysis as a clean,
run-reproducible pipeline while preserving the original model families and the
business insights the author cared about:

1. EDA + correlation analysis
2. Statistical modelling (OLS, VIF, chi-square)
3. SVM kernel comparison, KNN tuning, Decision Tree + SMOTE
4. KMeans segmentation (elbow → 5 clusters)
5. Rule-based "high-risk employee" filters

## 3. Architecture

```
src/
├── config.py            # paths, constants, seeds, model knobs
├── load_data.py         # dataset loading + column normalisation
├── preprocess.py        # cleaning, encoding, stratified train/test split
├── eda.py               # figures + driver summaries (output/figures/)
├── feature_analysis.py  # OLS, VIF, chi-square
├── models.py            # SVM / KNN / DecisionTree+SMOTE comparison
├── clustering.py        # KMeans elbow + profiling (output/clusters/)
├── risk_filters.py      # explainable rule-based risk alerts
└── run_all.py           # end-to-end entry point
```

Each module is independently runnable (`python models.py`, etc.) and
`run_all.py` orchestrates the full pipeline. All randomness is seeded
(`config.SEED = 42`), so results are reproducible.

## 4. Pipeline stages

### 4.1 Data loading & cleaning
- Strip whitespace from column names, normalise `Attrition` values to `Yes/No`.
- **Dropped no-signal columns**: `EmployeeNumber` (row id), `EmployeeCount`,
  `Over18`, `StandardHours` (all constant).

### 4.2 EDA
Recreates the useful charts from all six notebooks (univariate distributions
with skew/kurtosis, categorical counts split by attrition, attrition-rate per
factor, full correlation heatmap). Top correlates with attrition:

| Feature | |corr| with Attrition |
|---|---|---|
| OverTime | 0.246 |
| TotalWorkingYears | −0.171 |
| JobLevel | −0.169 |
| MaritalStatus | 0.162 |
| YearsInCurrentRole | −0.161 |
| MonthlyIncome | −0.160 |
| Age | −0.159 |
| YearsWithCurrManager | −0.156 |

### 4.3 Statistical analysis (feature_analysis.py)
Carried over from `Hr_attrition_last_action_SVM_final.ipynb`:

- **OLS regression** (R² = 0.21): strongest absolute coefficients are
  `OverTime` (0.21), `JobInvolvement`, `Department`, `MaritalStatus`,
  `EnvironmentSatisfaction`, `JobSatisfaction`. Interpretation is limited —
  OLS on a binary outcome is a linear-probability model, treat as exploratory.
- **VIF**: several features are heavily collinear (PerformanceRating ≈ 122,
  JobLevel ≈ 53, PercentSalaryHike ≈ 42, Age ≈ 33, MonthlyIncome ≈ 32);
  the salary/experience/job-level block and the rating/hike block are strongly
  coupled.
- **Chi-square**: retained for parity, but note that applying chi-square to
  continuous features (MonthlyIncome, rates) violates its assumptions; we treat
  it purely as a ranking heuristic.

### 4.4 Modelling (models.py)
Model families preserved from the original notebooks, with corrected
methodology (see §5):

| Model | Acc | Prec | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SVM (linear) | 0.779 | 0.390 | 0.681 | 0.496 | **0.804** |
| SVM (rbf) | 0.823 | 0.458 | 0.574 | **0.509** | 0.802 |
| SVM (poly) | 0.803 | 0.396 | 0.447 | 0.420 | 0.753 |
| SVM (sigmoid) | 0.721 | 0.323 | 0.681 | 0.438 | 0.752 |
| KNN (k=7) | **0.857** | 0.692 | 0.191 | 0.300 | 0.741 |
| DecisionTree + SMOTE | 0.741 | 0.301 | 0.468 | 0.367 | 0.663 |

**Takeaway**: no model is meaningfully better than predicting the majority
class on accuracy alone. Reported precision/recall/F1/ROC-AUC show the models
are *modestly* informative — the problem is intrinsically hard with this
feature set — and SVM with a linear/rbf kernel is the best family. Decision
Tree+SMOTE underperforms; the tree is over-regularised for the class imbalance.

### 4.5 Clustering (clustering.py)
Elbow on TWSS is not sharply cut; continuation with k=5 (as the original chose)
yields:

| Cluster | n | Attrition rate |
|---|---|---|
| 0 | 400 | 17.5% |
| 1 | 388 | 18.6% |
| 2 | 144 | **6.3%** |
| 3 | 146 | **9.6%** |
| 4 | 392 | 18.4% |

Clusters 2 and 3 are "low-attrition" segments worth characterising as a
retention playbook. Full enriched frame → `output/clusters/Kmeans_HR_Attrition.xlsx`.

### 4.6 Rule-based risk alerts (risk_filters.py)
Explainable rules from `HR target_EDA.ipynb`, applied to **currently employed
staff** (the population HR can act on):

- **Filter 1** — `OverTime=Yes AND MaritalStatus=Single`: **66 employees**
  (131 across the whole dataset).
- **Filter 2** — + `Laboratory Technician`, `Research & Development`, `Male`,
  `PerformanceRating=3`: **3 employees** (12 across the whole dataset).

> The original notebooks reported 132 and 6. The difference comes from
> dataset version (a "yogesh" variant) and inconsistent subpopulation
> filtering in the original code. The exact rule is the valuable artifact;
> the exact count depends on the rules' population scope.

## 5. Methodology corrections vs the original notebooks

1. **Local file paths** — replaced `/content/...` Colab paths.
2. **Data leakage** — the original feeding of `EmployeeNumber` (a unique row
   id) into classifiers produced memorized scores; and constants
   (`EmployeeCount`, `Over18`, `StandardHours`) polluted OLS/VIF. All dropped.
   (An intermediate version of this consolidation also leaked the integer
   target into the features — caught via a perfect 100% test score, and fixed;
   added as a regression check.)
3. **Scaler fitted on train only** — `MinMaxScaler` is fit on the training
   fold, then applied to test, instead of normalising the whole frame before
   splitting.
4. **Stratified split + fixed seed** — reproducible, class-preserving split.
5. **Imbalance-aware metrics** — the original headline "84.69% accuracy"
   (linear SVM / 29-NN) is barely above the 83.9% majority-class base rate;
   we report precision, recall, F1, ROC-AUC and confusion matrices instead of
   accuracy alone.
6. **One-hot encoding** — categoricals in the classifier pipeline are
   one-hot encoded instead of arbitrarily ordinal-encoded; label-encoding is
   retained only in the stats/correlation stages (mirroring original intent).
7. **Removed broken code** — the attempted Keras MLP (invalid `input_shape`,
   missing tensorflow/keras imports) is out of scope; listed in §7.

## 6. Key business insights

1. **Overtime work is the single strongest attrition driver** (correlation
   0.25, largest OLS coefficient, top chi-square categorical). Reducing
   systematic overtime is the cheapest retention lever.
2. **Tenure-protective factors**: TotalWorkingYears, JobLevel, YearsInCurrentRole,
   MonthlyIncome, YearsWithCurrManager all negatively correlate with attrition —
   i.e. seniority and income retention.
3. **Singles and frequent travellers leave more**; management/interventions
   should target lifestyle-fit and travel load.
4. **Segmentation works**: ~20% of the workforce (clusters 2–3) sits at
   <10% attrition; clusters 0/1/4 run ~18% attrition and deserve targeted
   retention programs.
5. **Explainable rules provide an immediately usable watchlist** (§4.6).

## 7. Limitations & future work

- OLS is used as a linear-probability model; a logistic regression (or
  penalised logistic) is the natural successor and gives odds-ratios for
  stakeholders.
- Chi-square on numeric columns is a heuristic only.
- Class imbalance left SVM via `class_weight="balanced"` and the tree via
  SMOTE; try SMOTE + tuned ensemble (Random Forest / Gradient Boosting),
  hyperparameter search over the SVM kernels, and ROC-PR thresholds tuned to a
  per-attrition-cost objective.
- Add cross-validation (stratified k-fold) instead of a single split, and
  report mean ± std.
- Cluster profiling with per-cluster feature medians to describe
  action-oriented segments.
- Subgroup/fairness checks (gender, department) before acting on "risk lists".
- Optional: implement the MLP properly or swap to a small
  `MLPClassifier`/`LightGBM`; `torch` was unavailable on this machine.

## 8. Reproducibility

```bash
pip install -r requirements.txt
cd src && python run_all.py
```

Requires `data/raw/Dataset - HR Employee Attrition.csv` (canonical IBM HR
Attrition CSV, 1,470 rows), included as downloaded. All outputs land under
`output/`; every random draw uses `SEED = 42`.