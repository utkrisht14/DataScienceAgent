"""Descriptive statistics and categorical frequency analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_scout.serialization import to_json_safe


def summarize_columns(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate a compact summary for selected columns."""
    selected_columns = columns or list(dataframe.columns)
    selected_columns = [
        column for column in selected_columns if column in dataframe.columns
    ][:30]

    if not selected_columns:
        raise ValueError("No valid columns were selected.")

    summaries: dict[str, Any] = {}

    for column in selected_columns:
        series = dataframe[column]
        summary: dict[str, Any] = {
            "dtype": str(series.dtype),
            "count": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            summary.update(
                {
                    "mean": clean.mean() if len(clean) else None,
                    "median": clean.median() if len(clean) else None,
                    "standard_deviation": clean.std() if len(clean) > 1 else None,
                    "minimum": clean.min() if len(clean) else None,
                    "first_quartile": clean.quantile(0.25) if len(clean) else None,
                    "third_quartile": clean.quantile(0.75) if len(clean) else None,
                    "maximum": clean.max() if len(clean) else None,
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            clean = series.dropna()
            summary.update(
                {
                    "earliest": clean.min() if len(clean) else None,
                    "latest": clean.max() if len(clean) else None,
                }
            )
        else:
            top_values = series.fillna("<missing>").astype(str).value_counts().head(5)
            summary["top_values"] = {
                str(key): int(value) for key, value in top_values.items()
            }

        summaries[column] = summary

    return to_json_safe({"column_summaries": summaries})


def get_value_counts(
    dataframe: pd.DataFrame,
    column: str,
    top_n: int = 10,
) -> dict[str, Any]:
    """Return the most frequent values in one column."""
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    top_n = max(1, min(int(top_n), 50))
    counts = (
        dataframe[column]
        .fillna("<missing>")
        .astype(str)
        .value_counts(dropna=False)
        .head(top_n)
    )

    total = max(len(dataframe), 1)
    rows = [
        {
            "value": str(value),
            "count": int(count),
            "percentage": round(float(count / total * 100), 2),
        }
        for value, count in counts.items()
    ]

    return {"column": column, "top_n": top_n, "values": rows}
