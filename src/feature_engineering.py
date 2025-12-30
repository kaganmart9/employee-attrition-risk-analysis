import pandas as pd
import numpy as np


def _safe_quartile_inference(series: pd.Series, edges: list[float]) -> pd.Series:
    """
    Assign quartile bucket (0-3) using fixed bin edges for inference.
    This avoids pd.qcut instability for single-row inputs.
    """
    # pd.cut returns NaN if value outside bounds unless we expand the edges.
    # We'll clamp by extending edges to -inf/+inf.
    bins = [-np.inf] + edges + [np.inf]  # edges should be the 3 internal cut points
    # That yields 5 bins; but we want 4 quartiles => edges should be 3 cut points.
    # Example: edges=[2911,4919,8379] => bins=[-inf,2911,4919,8379,inf] => 4 bins.
    out = pd.cut(series, bins=bins, labels=[0, 1, 2, 3], include_lowest=True)
    return out.astype("float").fillna(0).astype(int)


def add_feature_engineering(df: pd.DataFrame, mode: str = "train") -> pd.DataFrame:
    """
    Apply feature engineering.

    mode:
      - "train": uses qcut quartiles (data-driven within dataset)
      - "inference": uses fixed cut points to remain stable for single-row predictions
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Guardrails: ensure required base columns exist
    # ------------------------------------------------------------------
    required_cols = [
        "MonthlyIncome",
        "Age",
        "YearsAtCompany",
        "TotalWorkingYears",
        "DistanceFromHome",
        "BusinessTravel",
        "OverTime",
        "MaritalStatus",
        "JobInvolvement",
        "JobSatisfaction",
        "WorkLifeBalance",
        "EnvironmentSatisfaction",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature engineering input is missing required columns: {missing}"
        )

    # ------------------------------------------------------------------
    # Quartile features
    # ------------------------------------------------------------------
    if mode == "train":
        df["income_quartile"] = pd.qcut(
            df["MonthlyIncome"], 4, labels=False, duplicates="drop"
        )
        df["age_quartile"] = pd.qcut(df["Age"], 4, labels=False, duplicates="drop")
        df["tenure_quartile"] = pd.qcut(
            df["YearsAtCompany"], 4, labels=False, duplicates="drop"
        )
        df["exp_quartile"] = pd.qcut(
            df["TotalWorkingYears"], 4, labels=False, duplicates="drop"
        )
        df["distance_quartile"] = pd.qcut(
            df["DistanceFromHome"], 4, labels=False, duplicates="drop"
        )
    else:
        # Fixed cut points taken from your EDA bucket outputs
        # MonthlyIncome buckets: (1008.999, 2911], (2911, 4919], (4919, 8379], (8379, 19999]
        df["income_quartile"] = _safe_quartile_inference(
            df["MonthlyIncome"], edges=[2911.0, 4919.0, 8379.0]
        )

        # Age buckets: (18,30], (30,36], (36,43], (43,60]
        df["age_quartile"] = _safe_quartile_inference(
            df["Age"], edges=[30.0, 36.0, 43.0]
        )

        # YearsAtCompany buckets: (0,3], (3,5], (5,9], (9,40]
        df["tenure_quartile"] = _safe_quartile_inference(
            df["YearsAtCompany"], edges=[3.0, 5.0, 9.0]
        )

        # TotalWorkingYears buckets: (0,6], (6,10], (10,15], (15,40]
        df["exp_quartile"] = _safe_quartile_inference(
            df["TotalWorkingYears"], edges=[6.0, 10.0, 15.0]
        )

        # DistanceFromHome buckets: (1,2], (2,7], (7,14], (14,29]
        df["distance_quartile"] = _safe_quartile_inference(
            df["DistanceFromHome"], edges=[2.0, 7.0, 14.0]
        )

    # ------------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------------
    df["low_income_flag"] = (df["MonthlyIncome"] < 3000).astype(int)
    df["early_tenure_flag"] = (df["YearsAtCompany"] <= 1).astype(int)
    df["new_hire_flag"] = (df["TotalWorkingYears"] <= 1).astype(int)
    df["long_distance_flag"] = (df["DistanceFromHome"] >= 15).astype(int)

    df["frequent_travel_flag"] = (df["BusinessTravel"] == "Travel_Frequently").astype(
        int
    )
    df["overtime_flag"] = (df["OverTime"] == "Yes").astype(int)
    df["is_single_flag"] = (df["MaritalStatus"] == "Single").astype(int)

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------
    df["low_income_and_overtime"] = (
        df["low_income_flag"] & df["overtime_flag"]
    ).astype(int)
    df["early_tenure_and_overtime"] = (
        df["early_tenure_flag"] & df["overtime_flag"]
    ).astype(int)
    df["new_hire_and_overtime"] = (df["new_hire_flag"] & df["overtime_flag"]).astype(
        int
    )
    df["frequent_travel_and_overtime"] = (
        df["frequent_travel_flag"] & df["overtime_flag"]
    ).astype(int)

    # ------------------------------------------------------------------
    # Composite indices
    # ------------------------------------------------------------------
    df["travel_intensity"] = df["BusinessTravel"].map(
        {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    )

    df["engagement_index"] = (
        df["JobInvolvement"]
        + df["JobSatisfaction"]
        + df["WorkLifeBalance"]
        + df["EnvironmentSatisfaction"]
    ) / 4

    # ------------------------------------------------------------------
    # Final safety: inference should never output NaN
    # ------------------------------------------------------------------
    if df.isnull().any().any():
        # For this project, fill numeric NaNs with 0 (safe default).
        # Categorical NaNs also become 0, but we avoid NaN categories in app anyway.
        df = df.fillna(0)

    return df
