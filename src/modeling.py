"""Leakage-safe, reproducible model training and held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.clean_data import TARGET

RANDOM_STATE = 42
TEST_SIZE = 0.20
ID_COLUMN = "id"
CATEGORICAL_COLUMNS = [
    "sex", "education", "marriage", "pay_1", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"
]


@dataclass
class ModelResults:
    """Fitted model artifacts plus untouched held-out observations."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    models: dict[str, Pipeline]
    probabilities: dict[str, np.ndarray]
    predictions: dict[str, np.ndarray]


def split_features(data: pd.DataFrame):
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column {TARGET!r}")
    y = data[TARGET].astype(int)
    X = data.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    if ID_COLUMN in X.columns or TARGET in X.columns:
        raise AssertionError("Identifier or target leaked into predictors")
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def make_preprocessor(feature_columns, scale_numeric: bool = True) -> ColumnTransformer:
    categorical = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(feature_columns) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor(feature_columns, scale_numeric=True)),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor(feature_columns, scale_numeric=False)),
            ("classifier", RandomForestClassifier(
                n_estimators=160,
                max_depth=16,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


def evaluate_predictions(y_true, probabilities, threshold: float = 0.5) -> dict:
    """Calculate classification and probability metrics from supplied predictions."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": matrix,
    }


def train_models(data: pd.DataFrame) -> ModelResults:
    X_train, X_test, y_train, y_test = split_features(data)
    models = build_models(X_train.columns)
    probabilities, predictions = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities[name] = model.predict_proba(X_test)[:, 1]
        predictions[name] = (probabilities[name] >= 0.5).astype(int)
    return ModelResults(X_train, X_test, y_train, y_test, models, probabilities, predictions)
