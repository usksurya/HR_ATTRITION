# HR Employee Attrition Analysis

Consolidated, reproducible replacement for the six ad-hoc notebooks in this
repo. Same dataset, same model families, same insights - but with corrected
methodology and a single entry point.

## Quick start

```bash
pip install -r requirements.txt
cd src
python run_all.py
```

Artifacts:

| Stage | Output |
|---|---|
| EDA figures | `output/figures/*.png` |
| Statistical analysis (OLS, VIF, chi-square) | stdout |
| Model comparison | `output/models/model_comparison.csv` |
| KNN k-tuning grid | `output/models/knn_tuning.csv` |
| KMeans clusters + elbow plot | `output/clusters/` |
| Rule-based risk alerts | stdout |

## Dataset

`data/raw/Dataset - HR Employee Attrition.csv` - the IBM HR Analytics
attrition dataset (1,470 employees, 35 columns, no missing values). Downloaded
from IBM's public repo (see `DESIGN.md` for details).

## Original notebooks vs this project

| Original file | Content | Status |
|---|---|---|
| `AI - HR Attrition.ipynb` | everything (EDA + stats + SVM/KNN/DT + KMeans) | de-duplicated (duplicate of `AI_HR_Attrition.ipynb`) |
| `AI_HR_Attrition.ipynb` | same as above | replaced by `src/*` |
| `HR target_EDA.ipynb` | factor EDA + chi-square + risk filters | `eda.py`, `feature_analysis.py`, `risk_filters.py` |
| `Hr_attrition_last_action_SVM_final.ipynb` | EDA + OLS + VIF + SVM kernels | `feature_analysis.py`, `models.py` |
| `HR_Employee_Attrition.ipynb` | basic EDA + encoding | `eda.py`, `preprocess.py` |
| `Kiran_Surya_EDA.ipynb` | EDA + heatmaps | `eda.py` |

Read the full design in [`DESIGN.md`](DESIGN.md).