from typing import List, Tuple
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_groups(
    df: pd.DataFrame, target_col: str
) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numerical feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe including target column.
    target_col : str
        Name of the target column.

    Returns
    -------
    cat_cols : List[str]
        Categorical feature columns.
    num_cols : List[str]
        Numerical feature columns.
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    return cat_cols, num_cols


def build_preprocessor(
    df: pd.DataFrame,
    target_col: str,
) -> ColumnTransformer:
    """
    Build preprocessing ColumnTransformer:
    - StandardScaler for numerical features
    - OneHotEncoder for categorical features

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe including target.
    target_col : str
        Target column name.

    Returns
    -------
    preprocessor : ColumnTransformer
        Sklearn preprocessing transformer.
    """
    cat_cols, num_cols = get_feature_groups(df, target_col)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor
