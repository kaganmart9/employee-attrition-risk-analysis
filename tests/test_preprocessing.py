import sys
from pathlib import Path

# --- Fix "No module named src" ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# --------------------------------

import numpy as np
import pandas as pd

from src.data_loader import load_processed_data
from src.preprocessing import build_preprocessor


def test_preprocessing_output_shape():
    """
    Test that preprocessing pipeline:
    - fits without error
    - returns a 2D numpy array
    - preserves row count
    - produces at least 1 feature
    """
    df = load_processed_data("hr_employee_attrition_features.csv")

    target_col = "AttritionFlag"

    # Build preprocessor USING FULL DF (as designed)
    preprocessor = build_preprocessor(
        df=df,
        target_col=target_col,
    )

    # Split X / y AFTER building preprocessor
    X = df.drop(columns=[target_col])

    X_transformed = preprocessor.fit_transform(X)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] > 0


def test_preprocessing_feature_names():
    """
    Test that feature names can be extracted
    and are non-empty.
    """
    df = load_processed_data("hr_employee_attrition_features.csv")

    target_col = "AttritionFlag"

    preprocessor = build_preprocessor(
        df=df,
        target_col=target_col,
    )

    X = df.drop(columns=[target_col])

    preprocessor.fit(X)

    feature_names = preprocessor.get_feature_names_out()

    assert feature_names is not None
    assert len(feature_names) > 0
