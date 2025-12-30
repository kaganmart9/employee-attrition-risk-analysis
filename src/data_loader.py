from pathlib import Path
import pandas as pd

# Project root = src/../
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load raw (untouched) dataset.
    Used only in initial audit / ingestion steps.
    """
    path = DATA_DIR / "raw" / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)


def load_interim_data(filename: str) -> pd.DataFrame:
    """
    Load interim dataset (after audit/cleaning, before feature engineering).
    """
    path = DATA_DIR / "interim" / filename
    if not path.exists():
        raise FileNotFoundError(f"Interim data file not found: {path}")
    return pd.read_csv(path)


def load_processed_data(filename: str) -> pd.DataFrame:
    """
    Load processed dataset (after feature engineering).
    This is the ONLY data source for modeling, inference and tests.
    """
    path = DATA_DIR / "processed" / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")
    return pd.read_csv(path)
