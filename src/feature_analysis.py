"""Statistical feature analysis - carries over the two statistical analyses
from the original 'Hr_attrition_last_action_SVM_final.ipynb' notebook:

  1. OLS regression of Attrition on every feature (statsmodels).
     Flagged in the design doc: OLS assumes a continuous target, so treat the
     coefficients as exploratory signal, not causal estimates.

  2. VIF (Variance Inflation Factor) to expose multicollinearity.

  3. Chi-square feature importances (from 'HR target_EDA.ipynb').
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import chi2

from config import TARGET
from preprocess import drop_no_signal_columns, label_encode_all


def _cleaned(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode and drop no-signal columns for stats models."""
    return label_encode_all(drop_no_signal_columns(df))


def run_ols(df: pd.DataFrame) -> RegressionResultsWrapper:
    """OLS on label-encoded data (mirrors the original formula-based model)."""
    encoded = _cleaned(df)
    formula = TARGET + " ~ " + " + ".join(c for c in encoded.columns if c != TARGET)
    model = smf.ols(formula, data=encoded).fit()
    print("\n[OLS] summary (top coefficients):")
    coefs = model.params.abs().drop("Intercept").sort_values(ascending=False)
    print(f"[OLS] R-squared = {model.rsquared:.4f}")
    print(coefs.head(10).to_string())
    return model


def run_vif(df: pd.DataFrame) -> pd.DataFrame:
    """VIF for every feature after label encoding. VIF > 10 = collinear."""
    encoded = _cleaned(df).drop(columns=[TARGET])
    X = encoded.values.astype(float)
    vif_df = pd.DataFrame({"feature": encoded.columns, "VIF": 0.0})
    for i in range(X.shape[1]):
        vif_df.iloc[i, 1] = variance_inflation_factor(X, i)
    vif_df = vif_df.sort_values("VIF", ascending=False)
    print("\n[VIF] high-collinearity features (VIF > 10):")
    print(vif_df[vif_df["VIF"] > 10].to_string(index=False))
    return vif_df


def run_chi2(df: pd.DataFrame) -> pd.Series:
    """Chi-square scores of categorical features vs the target."""
    work = _cleaned(df)
    X = work.drop(columns=[TARGET])
    y = work[TARGET]
    scores, _ = chi2(X, y)
    chi_values = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    print("\n[CHI2] feature importance scores:")
    print(chi_values.to_string())
    return chi_values


def run_all(df: pd.DataFrame) -> None:
    run_ols(df)
    run_vif(df)
    run_chi2(df)