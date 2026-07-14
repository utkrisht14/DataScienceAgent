""" Foundational baseline model with sci-kit pipeline """

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_squared_error,
    mean_absolute_error, r2_score
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_scout.serialization import to_json_safe


def train_baseline_model(
        dataframe: pd.DataFrame,
        target: str,
        task_type: str = "auto",
        features: list[str] | None = None,
        test_size: float = 0.20,
        random_state: int = 42,
        ) -> dict[str, Any]:

    """Train and evaluate a Random Forest classification or regression model."""

    if target not in dataframe.columns:
        raise ValueError(f"Target column '{target}' not found in the dataframe.")
    if not 0.10 < test_size < 0.4:
        raise ValueError("Test size must be between 0.1 and 0.4.")

    working = dataframe.dropna(subset=[target]).copy()
    if len(working) < 30:
        raise ValueError("At least 30 rows with a non-missing target are required.")

    selected_features = _select_features(working, target, features)

    if not selected_features:
        raise ValueError("No usable feature remains after validation.")

    resolved_task = _resolve_task_type(working[target], task_type)

    x = working[selected_features].copy()
    y = working[target].copy()

    if resolved_task == "classification":
        class_count = int(y.nunique(dropna=True))
        if class_count < 2:
            raise ValueError("Classification requires at least two classes.")
        if class_count > 20:
            raise ValueError("Classification is not recommended for more than 20 classes.")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        random_state=random_state, stratify = y if resolved_task == "classification" else None)

    numeric_features = x.select_dtypes(include=np.number).columns.tolist()
    categorical_features = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    transformers =[]
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps = [("imputer" , SimpleImputer(strategy="median"))]
        )
        transformers.append(("numeric", numeric_pipeline))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps = [("imputer" , SimpleImputer(strategy="most_frequent")),
                     ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, max_categories=30,
                                               ),),
                     ]
        )
        transformers.append(("categorical", categorical_pipeline))

    if not transformers:
        raise ValueError("The selected features contain no supported data types.")

    preprocessor = ColumnTransformer(transformers=transformers)

    if resolved_task == "classification":
        estimator = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=random_state,
                                           class_weight="balanced", n_jobs=-1)
    else:
        estimator = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=random_state,
                                          n_jobs=-1)

    pipeline = Pipeline(
        steps = [("preprocessor", preprocessor),("model", estimator)]
    )

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics: dict[str, Any]
    if resolved_task == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "weighted_f1": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classes": [str(value) for value in sorted(y.astype(str).unique())],
        }
    else:
        mse = mean_squared_error(y_test, predictions)
        metrics  = {
            "mean_squared_error": mse,
            "root_mean_squared_error": float(np.sqrt(mse)),
            "mean_absolute_error": mean_absolute_error(y_test, predictions),
            "r_squared": r2_score(y_test, predictions),
        }

    feature_importance = _extract_feature_importance(pipeline)

    result = {
        "task_type": resolved_task,
        "model": type(estimator).__name__,
        "target": target,
        "features_requested": features,
        "features_used": selected_features,
        "rows_used": int(len(working)),
        "training_rows": int(len(x_train)),
        "testing_rows": int(len(x_test)),
        "metrics": metrics,
        "top_feature_importance": feature_importance,
        "limitations": [
            "This is a baseline educational model, not a production model.",
            "Feature importance shows predictive usefulness, not causation.",
            "Check for target leakage, sampling bias, and business validity.",
        ],
    }
    return to_json_safe(result)


def _resolve_task_type(target: pd.Series, requested: str) -> str:
    requested = requested.lower().strip()
    if requested in {"classification", "regression"}:
        return requested
    if requested != "auto":
        raise ValueError("task_type must be auto, classification, or regression.")

    # Numeric targets with many distinct values are treated as regression.
    if pd.api.types.is_numeric_dtype(target) and target.nunique() > 20:
        return "regression"
    return "classification"


def _select_features(
    dataframe: pd.DataFrame,
    target: str,
    requested_features: list[str] | None,
) -> list[str]:
    candidates = requested_features or [
        column for column in dataframe.columns if column != target
    ]
    candidates = [
        column
        for column in candidates
        if column in dataframe.columns and column != target
    ]

    selected: list[str] = []
    row_count = max(len(dataframe), 1)

    for column in candidates:
        series = dataframe[column]

        # Raw datetime columns are excluded to keep preprocessing simple. Users
        # can first create year/month features with the feature tool.
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        if pd.api.types.is_numeric_dtype(series):
            if series.nunique(dropna=True) > 1:
                selected.append(column)
            continue

        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(series):
            unique_count = int(series.nunique(dropna=True))
            unique_ratio = unique_count / row_count
            # Exclude likely identifiers and free-text columns.
            if 1 < unique_count <= 100 and unique_ratio <= 0.50:
                selected.append(column)

    return selected[:25]


def _extract_feature_importance(pipeline: Pipeline) -> list[dict[str, Any]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    try:
        names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_
    except (AttributeError, ValueError):
        return []

    pairs = sorted  (
        zip(names, importances, strict=False),
        key = lambda pair: pair[1],
        reverse = True
    )[:15]

    return [
        {"feature": str(name), "importance": float(importance)}
        for name, importance in pairs
    ]






