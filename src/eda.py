"""Exploratory Data Analysis - consolidates the EDA scattered across the
original six notebooks into one reproducible set of figures + summary stats.

Included (de-duplicated from original work):
  * missing-value check, data types, basic stats
  * univariate distributions (numerics) with skew/kurtosis
  * categorical plots against Attrition
  * target-rates by subgroup for the key drivers surfaced in the original
    'HR target_EDA.ipynb' (OverTime, MaritalStatus, Department, JobRole)
  * correlation heatmap of the fully encoded dataset
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES, TARGET
from preprocess import drop_no_signal_columns, label_encode_all, split_categorical

sns.set_theme(style="whitegrid")
HEAT_COLS = ["OverTime", "MaritalStatus", "Department", "JobRole", "Gender", "EducationField", "BusinessTravel"]


def figure(name: str, width: float, height: float):
    plt.figure(figsize=(width, height))
    return str(FIGURES / name)


def summarize(df: pd.DataFrame) -> dict:
    """Return key summary numbers used across the pipeline."""
    is_num = df.dtypes.apply(lambda t: pd.api.types.is_numeric_dtype(t))
    n_num = int(is_num.sum())
    n_cat = int(len(df.columns) - n_num)
    out = {
        "rows": len(df),
        "n_numeric": n_num,
        "n_categorical": n_cat,
        "missing_total": int(df.isna().sum().sum()),
        "attrition_rate": float((df[TARGET] == "Yes").mean()),
        "n_attrition": int((df[TARGET] == "Yes").sum()),
    }
    return out


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna().sum().rename("missing").reset_index()


def describe_target_rates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Attrition rate per category of a categorical column."""
    g = df.groupby(col)[TARGET].agg(["count", lambda s: (s == "Yes").mean()])
    g.columns = ["n", "attrition_rate"]
    return g.sort_values("attrition_rate", ascending=False)


def run_eda(df: pd.DataFrame) -> None:
    # 1. Missing values (none, as the original notebooks found)
    print("\n[EDA] missing values per column (all-zero expected):")
    print(missing_values(df).to_string(index=False))

    # 2. Univariate distributions for numeric columns, with skew/kurtosis
    df_work = drop_no_signal_columns(df)
    num, cat = split_categorical(df_work)

    for col in num.columns:
        n = df_work[col]
        print(f"[EDA] {col}: skew={n.skew():+.3f} kurt={n.kurt():+.3f}")
        ax = sns.histplot(x=col, data=df_work, kde=True, color="steelblue")
        ax.set_title(f"{col} - distribution")
        fig = ax.get_figure()
        fig.savefig(figure(f"dist_{col}.png", 8, 5), dpi=110, bbox_inches="tight")
        plt.close(fig)

    # 3. Categorical counts split by Attrition (subset)
    for col in HEAT_COLS:
        ax = sns.countplot(x=col, data=df_work, hue=TARGET)
        ax.set_title(f"{col} vs {TARGET}")
        ax.tick_params(axis="x", rotation=45)
        fig = ax.get_figure()
        fig.savefig(figure(f"count_{col}.png", 10, 5), dpi=110, bbox_inches="tight")
        plt.close(fig)

    # 4. Attrition rate by each categorical column (the driver analysis)
    for col in cat.columns:
        if col == TARGET:
            continue
        rates = describe_target_rates(df_work, col).reset_index()
        sns.barplot(x=col, y="attrition_rate", data=rates)
        plt.gca().set_title(f"{TARGET} rate by {col}")
        plt.gca().tick_params(axis="x", rotation=45)
        plt.savefig(figure(f"rate_{col}.png", 10, 5), dpi=110, bbox_inches="tight")
        plt.close()

    # 5. Correlation heatmap on fully label-encoded data
    encoded = label_encode_all(df_work)
    corr = encoded.corr(numeric_only=True)
    plt.figure(figsize=(16, 13))
    sns.heatmap(corr, cmap="coolwarm", annot=False, cbar=True)
    plt.title("Correlation heatmap - all features (label-encoded)")
    plt.tight_layout()
    plt.savefig(figure("corr_heatmap.png", 16, 13), dpi=110, bbox_inches="tight")
    plt.close()

    # 6. Correlations with target, sorted
    corr_with_target = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\n[EDA] Top-10 features by |correlation| with Attrition:")
    print(corr_with_target.head(10).to_string())

    return corr_with_target


if __name__ == "__main__":
    from load_data import load

    run_eda(load())