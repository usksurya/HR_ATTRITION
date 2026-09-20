"""Actionable risk filters - preserves the rule-based attrition alerts from
'HR target_EDA.ipynb' (applied to currently-employed staff, i.e. the population
HR can actually retain):

  Filter 1: OverTime=Yes + MaritalStatus=Single        -> ~132 employees
  Filter 2: + JobRole=Laboratory Technician
            + Department=Research & Development
            + Gender=Male + PerformanceRating=3          -> ~6 employees

These are deliberately readable, explainable business rules - complementary
to the black-box classifiers.
"""
import pandas as pd

from config import TARGET


def high_risk_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Currently-employed employees (Attrition == 'No') matching the risk rules.
    Mirrors the population the original notebooks targeted for retention."""
    employees = df[df[TARGET] == "No"]
    baseline = employees[(employees["OverTime"] == "Yes") & (employees["MaritalStatus"] == "Single")]
    highest = baseline[
        (baseline["JobRole"] == "Laboratory Technician")
        & (baseline["Department"] == "Research & Development")
        & (baseline["Gender"] == "Male")
        & (baseline["PerformanceRating"] == 3)
    ]
    return baseline, highest


def run_risk_filters(df: pd.DataFrame) -> None:
    all_overtime_single = df[(df["OverTime"] == "Yes") & (df["MaritalStatus"] == "Single")]
    baseline, highest = high_risk_filter(df)
    print(f"\n[FILTERS] Filter 1 (OverTime=Yes & Single, all employees): {len(all_overtime_single)}")
    print(f"[FILTERS] Filter 1 (OverTime=Yes & Single, currently employed): {len(baseline)}")
    print(f"[FILTERS] Filter 2 (lab technician, R&D, male, perf=3, currently employed): {len(highest)}")
    cols = ["EmployeeNumber", TARGET, "OverTime", "MaritalStatus", "JobRole", "Department", "Gender", "PerformanceRating", "JobSatisfaction", "MonthlyIncome"]
    exist = [c for c in cols if c in highest.columns]
    if len(highest):
        print("[FILTERS] Highest-risk employees:")
        print(highest[exist].to_string(index=False))
    return all_overtime_single, baseline, highest


if __name__ == "__main__":
    from load_data import load

    run_risk_filters(load())