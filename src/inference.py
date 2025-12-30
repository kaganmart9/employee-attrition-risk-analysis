from pathlib import Path
import json
import joblib
import pandas as pd
from typing import Union, Dict


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def load_model(model_path: Union[str, Path]):
    """
    Load trained sklearn pipeline.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def load_model_config(config_path: Union[str, Path]) -> Dict:
    """
    Load lightweight model configuration (threshold, metadata).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def predict_proba(
    model,
    df: pd.DataFrame,
) -> pd.Series:
    """
    Predict attrition probability.
    """
    proba = model.predict_proba(df)[:, 1]
    return pd.Series(proba, index=df.index, name="attrition_proba")


def predict_with_threshold(
    model,
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Predict attrition probability and binary decision using threshold.
    """
    proba = predict_proba(model, df)
    prediction = (proba >= threshold).astype(int)

    return pd.DataFrame(
        {
            "attrition_proba": proba,
            "attrition_prediction": prediction,
        },
        index=df.index,
    )


def load_and_predict(
    df: pd.DataFrame,
    model_filename: str,
    config_filename: str,
) -> pd.DataFrame:
    """
    High-level helper:
    - load model
    - load config
    - run inference
    """
    model = load_model(MODELS_DIR / model_filename)
    config = load_model_config(MODELS_DIR / config_filename)

    threshold = config.get("threshold", 0.5)

    return predict_with_threshold(model, df, threshold)
