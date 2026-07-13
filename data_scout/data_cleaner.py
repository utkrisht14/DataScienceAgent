"""
Missing-value handling, duplicate removal, and conservative type conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

NUMERIC_STRATEGIES = {"median", "mode", "zero"}
CATEGORICAL_SERIES = {"mode", "unknown"}


def clean_dataframe(
        dataframe: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        convert_data_types: bool = True,
        remove_duplicates: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:

        """
        Returned a clean copy of the dataframe and a dictionary of the changes made.
        """
        if numeric_strategy not in NUMERIC_STRATEGIES:
            raise ValueError(f"Invalid numeric strategy: {numeric_strategy}")
        if categorical_strategy not in CATEGORICAL_SERIES:
            raise ValueError(f"Invalid categorical strategy: {categorical_strategy}")

        cleaned = dataframe.copy()
        actions: list[str] = []
        missing_before = int(cleaned.isna().sum().sum()) # Keep track of missing values before cleaning
        duplicates_before = int(cleaned.duplicated().sum())

        if remove_duplicates and duplicates_before:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            actions.append(f"Removed {duplicates_before} duplicate rows.")

        if convert_data_types:
            actions.extend(_convert_obvious_data_types(cleaned))

        numeric_columns =cleaned.select_dtypes(include=np.number).columns
        categorical_columns = cleaned.select_dtypes(
            include = ["object", "category", "bool"]
        ).columns

        for column in numeric_columns:
            missing_count = int(cleaned[column].isna().sum())
            if missing_count == 0:
                continue

            if numeric_strategy == "mean":
                fill_value = cleaned[column].mean()
            elif numeric_strategy == "zero":
                fill_value = 0
            else:
                fill_value = cleaned[column].median()

            # A completely empty numeric column has no mean or median
            if pd.isna(fill_value):
                fill_value = 0

            cleaned[column].fillna(fill_value, inplace=True)
            actions.append(
                f"Filled {missing_count} missing values in '{column}' "
                f"using {numeric_strategy}."
            )

        # Missing datetime values are intentionally left as NaT. Inventing a date
        # would usually be more misleading than keeping it missing.
        datetime_columns = cleaned.select_dtypes(
            include=["datetime", "datetimetz"]
        ).columns
        for column in datetime_columns:
            remaining = int(cleaned[column].isna().sum())
            if remaining:
                actions.append(
                    f"Left {remaining} missing datetime values in '{column}' unchanged."
                )

        report = {
            "shape_before": [int(dataframe.shape[0]), int(dataframe.shape[1])],
            "shape_after": [int(cleaned.shape[0]), int(cleaned.shape[1])],
            "missing_before": missing_before,
            "missing_after": int(cleaned.isna().sum().sum()),
            "duplicates_before": duplicates_before,
            "duplicates_after": int(cleaned.duplicated().sum()),
            "data_types_after": {
                column: str(dtype) for column, dtype in cleaned.dtypes.items()
            },
            "actions": actions,
        }
        return cleaned, report


def _convert_obvious_data_types(dataframe: pd.DataFrame) -> list[str]:
    """ Convert object columns when the columns are highly reliable """
    actions: list[str] = []

    for column in dataframe.columns:
        series = dataframe[column]
        if series.dtype != "object":
            continue

        non_null_count = series.notna().sum()
        if non_null_count == 0:
            continue

        lower_name = column.lower()
        date_name = any(
            token in lower_name
            for token in ("date", "time", "timestamp")
        )

        if date_name:
            parsed_dates = pd.to_datetime(series, errors="coerce", format="mixed")
            success_rate = parsed_dates.notna().sum() / non_null_count
            if success_rate > 0.8:
                dataframe[column] = parsed_dates
                actions.append(
                    f"Converted column '{column}' to datetime64[ns]."
                    f" Success rate: {success_rate:.2% of non-missing values parsed}"
                )
                continue

        parsed_numeric = pd.to_numeric(series, errors="coerce")
        success_rate = parsed_numeric.notna().sum() / non_null_count
        if success_rate > 0.95:
            dataframe[column] = parsed_numeric
            actions.append(
                f"Converted column '{column}' to numeric."
                f" Success rate: {success_rate:.2% of non-missing values parsed}"
            )

    return actions