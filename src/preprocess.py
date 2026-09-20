"""Preprocessing: cleaning, encoding and scaling.

Two pipelines are produced:

1. ``preprocess_analysis`` : LabelEncoder on every categorical column.
   Same approach as the original notebooks - used for the OLS regression,
   VIF check and correlation heatmaps (needs fully numeric data).

2. ``preprocess_modeling``   : a safe train/test split WITH a scaler fitted
   only on the training fold (no leakage), used for the classifiers.

A ~16.8% positive-attrition rate means the data is imbalanced, so the
modeling pipeline exposes two variants: raw (as-collected) and SMOTE
(resampled for the Decision Tree, matching the original work).
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from config import DROP_COLUMNS, SEED, TEST_SIZE, TARGET


def drop_no_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove constant / identifier columns that carry no predictive signal."""
    missing = [c for c in DROP_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns missing from dataset: {missing}")
    return df.drop(columns=DROP_COLUMNS)


def split_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into numeric and object/categorical columns."""
    num = df.select_dtypes(include="number")
    cat = df.select_dtypes(include=["object", "category"])
    return num, cat


def label_encode_all(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode every categorical column in place copy."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "category"]).columns:
        out[col] = LabelEncoder().fit_transform(out[col].astype(str))
    return out


def get_feature_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """Return a modelling-ready frame with the raw categorical target."""
    return df.copy()


def train_test_split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split of the *frame* (as in the original notebooks)."""
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df[TARGET],
    )
    return train, test


def build_modeling_pipeline(df: pd.DataFrame):
    """Encode target + one-hot/dummy categoricals, then scale on train only.

    Returns
    -------
    (X_train, X_test, y_train, y_test) all numeric, 0/1 target.
    """
    work = drop_no_signal_columns(df.copy())
    # Binary target out of the feature matrix
    y = (work[TARGET] == "Yes").astype(int).to_numpy()
    work = work.drop(columns=[TARGET])

    # One-hot encode categoricals (no ordering assumptions, no arbitrary ints).
    num, cat = split_categorical(work)
    base = num.copy()
    encoded = pd.get_dummies(cat, drop_first=False).astype(int)
    X = pd.concat([base, encoded], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # Scale numeric columns on TRAIN only, then transform test.
    scaler = MinMaxScaler()
    numeric_cols = [c for c in X_train.columns if c in num.columns]
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, scaler