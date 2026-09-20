"""Loading utilities for the HR Attrition dataset."""
import pandas as pd

from config import DATA_RAW_FILE, TARGET


def load_raw() -> pd.DataFrame:
    """Load the IBM HR Analytics attrition dataset and normalise column names."""
    if not DATA_RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_RAW_FILE}. "
            "Download it first (see README) or check the data/raw folder."
        )
    df = pd.read_csv(DATA_RAW_FILE)
    df.columns = [c.strip() for c in df.columns]
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title()
    return df


def load() -> pd.DataFrame:
    """Load the dataset ready for analysis (target stays categorical)."""
    return load_raw()