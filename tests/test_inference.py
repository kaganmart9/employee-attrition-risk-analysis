import sys
from pathlib import Path

# ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data_loader import load_processed_data
from src.inference import load_and_predict


def test_inference_output_columns():
    df = load_processed_data("hr_employee_attrition_features.csv")
    X = df.drop(columns=["AttritionFlag"])

    preds = load_and_predict(
        df=X,
        model_filename="logreg_l1_pipeline.joblib",
        config_filename="model_config.json",
    )

    assert "attrition_proba" in preds.columns
    assert "attrition_prediction" in preds.columns


def test_inference_probability_range():
    df = load_processed_data("hr_employee_attrition_features.csv")
    X = df.drop(columns=["AttritionFlag"])

    preds = load_and_predict(
        df=X,
        model_filename="logreg_l1_pipeline.joblib",
        config_filename="model_config.json",
    )

    assert preds["attrition_proba"].between(0, 1).all()
