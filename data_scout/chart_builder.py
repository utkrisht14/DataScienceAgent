"""Approved visualizations for exploratory data analysis.

The OpenAI model selects a chart type and column names, but it never writes or
executes arbitrary plotting code.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # Required for server-side chart creation.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


ALLOWED_PLOTS = {"histogram", "count", "scatter", "box", "correlation"}


def build_chart(
    dataframe: pd.DataFrame,
    plot_type: str,
    x: str | None = None,
    y: str | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Build one chart and return the figure plus compact metadata."""
    plot_type = plot_type.lower().strip()
    if plot_type not in ALLOWED_PLOTS:
        raise ValueError(
            f"Unsupported plot type '{plot_type}'. Allowed: {sorted(ALLOWED_PLOTS)}"
        )

    figure, axis = plt.subplots(figsize=(9, 5))

    if plot_type == "histogram":
        _require_numeric_column(dataframe, x, "x")
        sns.histplot(data=dataframe, x=x, kde=True, ax=axis)
        default_title = f"Distribution of {x}"

    elif plot_type == "count":
        _require_column(dataframe, x, "x")
        counts = (
            dataframe[x]
            .fillna("<missing>")
            .astype(str)
            .value_counts()
            .head(20)
            .sort_values()
        )
        # Matplotlib is used here because some Seaborn/Matplotlib version
        # combinations have a countplot color compatibility issue.
        axis.barh(counts.index, counts.values)
        axis.set_xlabel("Count")
        axis.set_ylabel(x)
        default_title = f"Most common values in {x}"

    elif plot_type == "scatter":
        _require_numeric_column(dataframe, x, "x")
        _require_numeric_column(dataframe, y, "y")
        sns.scatterplot(data=dataframe, x=x, y=y, ax=axis, alpha=0.65)
        default_title = f"{y} versus {x}"

    elif plot_type == "box":
        _require_numeric_column(dataframe, y, "y")
        if x is not None:
            _require_column(dataframe, x, "x")
            temporary = dataframe.copy()
            top_groups = (
                temporary[x]
                .fillna("<missing>")
                .astype(str)
                .value_counts()
                .head(12)
                .index
            )
            temporary[x] = temporary[x].fillna("<missing>").astype(str)
            temporary = temporary[temporary[x].isin(top_groups)]
            sns.boxplot(data=temporary, x=x, y=y, ax=axis)
            axis.tick_params(axis="x", rotation=35)
            default_title = f"Distribution of {y} by {x}"
        else:
            sns.boxplot(data=dataframe, y=y, ax=axis)
            default_title = f"Box plot of {y}"

    else:  # correlation
        numeric = dataframe.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            plt.close(figure)
            raise ValueError(
                "A correlation heatmap requires at least two numeric columns."
            )
        numeric = numeric.iloc[:, :20]
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=axis)
        default_title = "Numeric correlation heatmap"

    final_title = title or default_title
    axis.set_title(final_title)
    figure.tight_layout()

    metadata = {
        "plot_type": plot_type,
        "x": x,
        "y": y,
        "title": final_title,
        "rows_available": int(len(dataframe)),
    }
    return figure, metadata


def _require_column(
    dataframe: pd.DataFrame,
    column: str | None,
    argument_name: str,
) -> None:
    if not column:
        raise ValueError(f"The '{argument_name}' column is required.")
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' does not exist.")


def _require_numeric_column(
    dataframe: pd.DataFrame,
    column: str | None,
    argument_name: str,
) -> None:
    _require_column(dataframe, column, argument_name)
    assert column is not None
    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise ValueError(f"Column '{column}' must be numeric.")
