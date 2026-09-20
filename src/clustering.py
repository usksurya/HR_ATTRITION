"""KMeans clustering - reproduces the elbow/scree analysis and the 5-cluster
solution from 'AI_HR_Attrition.ipynb', writes the enriched frame to Excel
(kept for parity with the original `Kmeans_HR_Attrition.xlsx`).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from config import KMEANS_K_RANGE, KMEANS_N_CLUSTERS, CLUSTERS, TARGET
from preprocess import drop_no_signal_columns, label_encode_all


def elbow_curve(encoded: pd.DataFrame) -> int:
    """TWSS elbow plot; returns the selected number of clusters."""
    twss = []
    for k in KMEANS_K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(encoded)
        twss.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(KMEANS_K_RANGE), twss, "ro-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Total within-cluster sum of squares (inertia)")
    plt.title("KMeans elbow / scree plot")
    plt.savefig(CLUSTERS / "elbow.png", dpi=110, bbox_inches="tight")
    plt.close()
    return KMEANS_N_CLUSTERS


def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    encoded = drop_no_signal_columns(label_encode_all(df))
    n = elbow_curve(encoded)

    model = KMeans(n_clusters=n, n_init=10, random_state=42)
    labels = model.fit_predict(encoded)

    enriched = df.copy()
    enriched["Cluster"] = labels
    out_file = CLUSTERS / "Kmeans_HR_Attrition.xlsx"
    enriched.to_excel(out_file, index=False)

    profile = enriched.groupby("Cluster")[TARGET].agg(
        n="count",
        attrition_rate=lambda s: (s == "Yes").mean(),
    )
    print(f"\n[CLUSTERING] profiles for k={n}:")
    print(profile.to_string())
    print(f"[CLUSTERING] saved -> {out_file}")
    return enriched


if __name__ == "__main__":
    from load_data import load

    run_clustering(load())