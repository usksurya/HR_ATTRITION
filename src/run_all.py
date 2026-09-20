"""End-to-end entry point: load -> EDA -> stats -> models -> clustering -> rules.

Run from src/:

    python run_all.py

Each stage writes its artifacts (figures / model summaries / Excel) under
output/ and prints a concise summary to stdout.
"""
import json

import numpy as np
import pandas as pd

from config import ensure_dirs, FIGURES, MODELS

import eda
import feature_analysis
import models
import clustering
import risk_filters
from load_data import load

# Quiet down sktime/extraneous warnings from statsmodels
import warnings
warnings.filterwarnings("ignore")


def main() -> None:
    ensure_dirs()
    df = load()

    # 1. Basic dataset facts
    summary = eda.summarize(df)
    print("=== HR ATTRITION - dataset summary ===")
    print(json.dumps(summary, indent=2))

    # 2. EDA (figures into output/figures, drivers into stdout)
    eda.run_eda(df)

    # 3. Statistical analysis (OLS, VIF, chi-square)
    feature_analysis.run_all(df)

    # 4. Classifiers
    model_summary = models.run_all(df)
    model_summary["results"].to_csv(MODELS / "model_comparison.csv", index=False)
    model_summary["knn_grid"].to_csv(MODELS / "knn_tuning.csv", index=False)
    print(f"[MODELS] comparison saved -> {MODELS}/model_comparison.csv")
    print(f"[MODELS] best K = {model_summary['best_k']}")

    # 5. Clustering
    clustering.run_clustering(df)

    # 6. Rule-based risk filters
    risk_filters.run_risk_filters(df)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()