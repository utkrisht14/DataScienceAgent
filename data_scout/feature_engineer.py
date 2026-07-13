"""
Controlled feature engineering.

Only predefined operations are permitted. This is safer and easier to debug
than asking an AI model to generate and execute arbitrary Python code.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd

from data_scout.serialization import dataframe_preview

ALLOWED_OPERATIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "year",
    "month",
    "text_length"
}

def add_engineered_feature(
        dataframe: pd.DataFrame,
        operation: str,
        feature1: str,
        new_name: str,
        feature2: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:

    """Create one new feature and return a modified copy plus a report."""
    operation = operation.lower().strip()
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(
            f"Unsupported operation '{operation}'. Allowed: {sorted(ALLOWED_OPERATIONS)}"
        )
    if feature1 not in dataframe.columns:
        raise ValueError(f"Column '{feature1}' does not exist.")
    if not new_name.strip():
        raise ValueError("The new feature name cannot be empty.")
    if new_name in dataframe.columns:
        raise ValueError(f"Column '{new_name}' already exists.")

    result = dataframe.copy(deep=True)

    if operation in {"add", "subtract", "multiply", "divide"}:
        if not feature2 or feature2 not in dataframe.columns:
            raise ValueError("A valid feature2 column is required for this operation.")

        _require_numeric(result, feature1)
        _require_numeric(result, feature2)

        if operation == "add":
            result[new_name] = result[feature1] + result[feature2]
        elif operation == "subtract":
            result[new_name] = result[feature1] - result[feature2]
        elif operation == "multiply":
            result[new_name] = result[feature1] * result[feature2]
        else:
            denominator = result[feature2].replace(0, np.nan)
            result[new_name] = result[feature1] / denominator


    elif operation in {"year", "month"}:
        parsed = pd.to_datetime(result[feature1], errors="coerce", format="mixed")
        result[new_name] = parsed.dt.year if operation == "year" else parsed.dt.month

    else: # text-length
        result[new_name] = result[feature1].fillna("").astype(str).str.len()

    report = {
        "operation": operation,
        "source_columns": [
            column for column in (feature1, feature2) if column is not None
        ],
        "new_column": new_name,
        "new_dtype": str(result[new_name].dtype),
        "missing_values_in_new_column": int(result[new_name].isna().sum()),
        "preview": dataframe_preview(result[[new_name]], rows=5),
    }
    return result, report


def _require_numeric(dataframe: pd.DataFrame, column: str) -> None:
    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise ValueError(f"Column '{column}' must be numeric for this operation.")