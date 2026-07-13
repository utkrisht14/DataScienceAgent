"""Statistical hypothesis tests supported by Data Scout."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from data_scout.serialization import to_json_safe


ALLOWED_TESTS = {"t_test", "chi_square", "pearson"}


def run_hypothesis_test(
    dataframe: pd.DataFrame,
    test_type: str,
    numeric_column: str | None = None,
    group_column: str | None = None,
    group_a: str | None = None,
    group_b: str | None = None,
    column_x: str | None = None,
    column_y: str | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run one supported test and return a plain-language interpretation."""
    test_type = test_type.lower().strip()
    if test_type not in ALLOWED_TESTS:
        raise ValueError(
            f"Unsupported test '{test_type}'. Allowed: {sorted(ALLOWED_TESTS)}"
        )
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if test_type == "t_test":
        return _welch_t_test(
            dataframe,
            numeric_column=numeric_column,
            group_column=group_column,
            group_a=group_a,
            group_b=group_b,
            alpha=alpha,
        )

    if test_type == "chi_square":
        return _chi_square_test(
            dataframe,
            column_x=column_x,
            column_y=column_y,
            alpha=alpha,
        )

    return _pearson_test(
        dataframe,
        column_x=column_x,
        column_y=column_y,
        alpha=alpha,
    )


def _welch_t_test(
    dataframe: pd.DataFrame,
    numeric_column: str | None,
    group_column: str | None,
    group_a: str | None,
    group_b: str | None,
    alpha: float,
) -> dict[str, Any]:
    _require_columns(dataframe, [numeric_column, group_column])
    assert numeric_column is not None and group_column is not None

    if not pd.api.types.is_numeric_dtype(dataframe[numeric_column]):
        raise ValueError(f"'{numeric_column}' must be numeric for a t-test.")

    available_groups = dataframe[group_column].dropna().astype(str).unique().tolist()
    if group_a is None or group_b is None:
        if len(available_groups) != 2:
            raise ValueError(
                "Specify group_a and group_b when the group column does not "
                "contain exactly two groups."
            )
        group_a, group_b = available_groups

    group_text = dataframe[group_column].astype(str)
    values_a = dataframe.loc[group_text == str(group_a), numeric_column].dropna()
    values_b = dataframe.loc[group_text == str(group_b), numeric_column].dropna()

    if len(values_a) < 2 or len(values_b) < 2:
        raise ValueError("Each t-test group must contain at least two values.")

    statistic, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
    significant = bool(p_value < alpha)

    result = {
        "test": "Welch independent t-test",
        "null_hypothesis": (
            f"The mean of '{numeric_column}' is equal for {group_a} and {group_b}."
        ),
        "numeric_column": numeric_column,
        "group_column": group_column,
        "group_a": {
            "name": str(group_a),
            "sample_size": int(len(values_a)),
            "mean": values_a.mean(),
        },
        "group_b": {
            "name": str(group_b),
            "sample_size": int(len(values_b)),
            "mean": values_b.mean(),
        },
        "test_statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "statistically_significant": significant,
        "interpretation": (
            "Reject the null hypothesis; the group means differ significantly."
            if significant
            else "Do not reject the null hypothesis; the evidence is insufficient "
            "to conclude that the group means differ."
        ),
    }
    return to_json_safe(result)


def _chi_square_test(
    dataframe: pd.DataFrame,
    column_x: str | None,
    column_y: str | None,
    alpha: float,
) -> dict[str, Any]:
    _require_columns(dataframe, [column_x, column_y])
    assert column_x is not None and column_y is not None

    contingency = pd.crosstab(dataframe[column_x], dataframe[column_y])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise ValueError("Chi-square requires at least a 2 by 2 contingency table.")

    statistic, p_value, degrees_of_freedom, expected = stats.chi2_contingency(
        contingency
    )
    significant = bool(p_value < alpha)
    low_expected_cells = int((expected < 5).sum())

    result = {
        "test": "Chi-square test of independence",
        "null_hypothesis": f"'{column_x}' and '{column_y}' are independent.",
        "table_shape": [int(contingency.shape[0]), int(contingency.shape[1])],
        "test_statistic": statistic,
        "degrees_of_freedom": int(degrees_of_freedom),
        "p_value": p_value,
        "alpha": alpha,
        "statistically_significant": significant,
        "cells_with_expected_count_below_5": low_expected_cells,
        "assumption_warning": (
            "Some expected counts are below 5; interpret the result cautiously."
            if low_expected_cells
            else None
        ),
        "interpretation": (
            "Reject the null hypothesis; the variables appear associated."
            if significant
            else "Do not reject the null hypothesis; no statistically significant "
            "association was detected."
        ),
    }
    return to_json_safe(result)


def _pearson_test(
    dataframe: pd.DataFrame,
    column_x: str | None,
    column_y: str | None,
    alpha: float,
) -> dict[str, Any]:
    _require_columns(dataframe, [column_x, column_y])
    assert column_x is not None and column_y is not None

    for column in (column_x, column_y):
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise ValueError(f"'{column}' must be numeric for Pearson correlation.")

    paired = dataframe[[column_x, column_y]].dropna()
    if len(paired) < 3:
        raise ValueError("Pearson correlation requires at least three paired values.")
    if paired[column_x].nunique() < 2 or paired[column_y].nunique() < 2:
        raise ValueError("Pearson correlation requires variation in both columns.")

    coefficient, p_value = stats.pearsonr(paired[column_x], paired[column_y])
    significant = bool(p_value < alpha)

    result = {
        "test": "Pearson correlation test",
        "null_hypothesis": f"'{column_x}' and '{column_y}' have zero linear correlation.",
        "sample_size": int(len(paired)),
        "correlation_coefficient": coefficient,
        "p_value": p_value,
        "alpha": alpha,
        "statistically_significant": significant,
        "interpretation": (
            "The linear correlation is statistically significant."
            if significant
            else "The evidence is insufficient to conclude that a linear "
            "correlation exists."
        ),
    }
    return to_json_safe(result)


def _require_columns(
    dataframe: pd.DataFrame,
    columns: list[str | None],
) -> None:
    for column in columns:
        if not column:
            raise ValueError("A required column argument was not provided.")
        if column not in dataframe.columns:
            raise ValueError(f"Column '{column}' does not exist.")
