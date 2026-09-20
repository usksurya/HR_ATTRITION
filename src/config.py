"""Central configuration for the HR Attrition pipeline.

Every module in this project imports paths and tuning constants from here so
the pipeline stays consistent and reproducible.
"""
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_RAW = ROOT / "data" / "raw"
DATA_RAW_FILE = DATA_RAW / "Dataset - HR Employee Attrition.csv"

# Outputs
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
MODELS = OUTPUT / "models"
CLUSTERS = OUTPUT / "clusters"

# Columns dropped before any analysis. They carry no signal:
#   - EmployeeNumber  : row id / primary key
#   - EmployeeCount   : constant (always 1)
#   - Over18          : constant (always "Y")
#   - StandardHours   : constant (always 80)
DROP_COLUMNS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]

# Target column
TARGET = "Attrition"

# Reproducibility
SEED = 42
TEST_SIZE = 0.2

# Model knobs carried over from the original notebooks
SVM_KERNELS = ["linear", "rbf", "poly", "sigmoid"]
KNN_K_RANGE = range(3, 50, 2)
KMEANS_K_RANGE = range(2, 20)
KMEANS_N_CLUSTERS = 5
DT_PARAMS = {
    "criterion": "entropy",
    "max_depth": 5,
    "min_samples_split": 2,
    "random_state": SEED,
}


def ensure_dirs() -> None:
    for d in (FIGURES, MODELS, CLUSTERS, DATA_RAW):
        d.mkdir(parents=True, exist_ok=True)