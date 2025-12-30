from typing import Dict, Tuple, Literal
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


ModelType = Literal["logreg", "logreg_l1", "rf", "hgb"]


def build_model(model_type: ModelType, random_state: int = 42):
    """
    Build model object based on model_type.
    """
    if model_type == "logreg":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        )

    if model_type == "logreg_l1":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            penalty="l1",
            random_state=random_state,
        )

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=random_state,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def build_pipeline(preprocessor, model_type: ModelType) -> Pipeline:
    """
    Build full sklearn Pipeline: preprocessing + model.
    """
    model = build_model(model_type)

    pipe = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    return pipe


def cross_validate_model(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Dict[str, Tuple[float, float]]:
    """
    Perform stratified cross-validation and return mean ± std metrics.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "roc_auc": "roc_auc",
        "recall": "recall",
        "precision": "precision",
    }

    cv_results = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {}
    for metric in scoring.keys():
        mean = np.mean(cv_results[f"test_{metric}"])
        std = np.std(cv_results[f"test_{metric}"])
        summary[metric] = (mean, std)

    return summary


def fit_pipeline(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Fit pipeline on training data.
    """
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_on_test(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Evaluate fitted pipeline on test set with custom threshold.
    """
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    results = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return results
