import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_interim_data
from src.feature_engineering import add_feature_engineering


def test_feature_engineering_adds_features():
    df = load_interim_data("hr_employee_attrition_audit_clean.csv")
    df_fe = add_feature_engineering(df)

    assert df_fe.shape[1] > df.shape[1]
    assert "AttritionFlag" in df_fe.columns
