import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest
from src.data_loader import load_interim_data


def test_load_interim_data_success():
    df = load_interim_data("hr_employee_attrition_audit_clean.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0


def test_load_interim_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_interim_data("non_existing_file.csv")
