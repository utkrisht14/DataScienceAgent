"""Stateful coordinator for all local data-science operations.

Low-level modules work only with DataFrames. This engine owns the current
working DataFrame, keeps an activity log, stores generated chart images, and
calls those low-level functions.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from data_scout.chart_builder import build_chart
from data_scout.data_cleaner import clean_dataframe
from data_scout.dataset_inspector import build_agent_context, inspect_dataset
from data_scout.feature_engineer import add_engineered_feature
from data_scout.hypothesis_tester import run_hypothesis_test
from data_scout.model_trainer import train_baseline_model
from data_scout.statistics_analyzer import get_value_counts, summarize_columns


@dataclass
class AnalysisEngine:
    """Hold application state and expose safe operations to the AI agent."""

    dataframe: pd.DataFrame
    original_dataframe: pd.DataFrame = field(init=False)
    activity_log: list[dict[str, Any]] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    last_model_result: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.original_dataframe = self.dataframe.copy(deep=True)
        self.dataframe = self.dataframe.copy(deep=True)

    def reset(self) -> dict[str, Any]:
        """Restore the originally uploaded dataset and clear generated results."""
        self.dataframe = self.original_dataframe.copy(deep=True)
        self.activity_log.clear()
        self.charts.clear()
        self.last_model_result = None
        return {"status": "reset", "shape": list(self.dataframe.shape)}

    def agent_context(self) -> str:
        """Return compact context used in the OpenAI instructions."""
        return build_agent_context(self.dataframe)

    def inspect_dataset(self) -> dict[str, Any]:
        result = inspect_dataset(self.dataframe)
        return self._record("inspect_dataset", result)

    def clean_dataset(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        convert_data_types: bool = True,
        remove_duplicates: bool = True,
    ) -> dict[str, Any]:
        self.dataframe, result = clean_dataframe(
            self.dataframe,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            convert_data_types=convert_data_types,
            remove_duplicates=remove_duplicates,
        )
        return self._record("clean_dataset", result)

    def summarize_columns(
        self,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        result = summarize_columns(self.dataframe, columns=columns)
        return self._record("summarize_columns", result)

    def value_counts(self, column: str, top_n: int = 10) -> dict[str, Any]:
        result = get_value_counts(self.dataframe, column=column, top_n=top_n)
        return self._record("value_counts", result)

    def create_chart(
        self,
        plot_type: str,
        x: str | None = None,
        y: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        figure, metadata = build_chart(
            self.dataframe,
            plot_type=plot_type,
            x=x,
            y=y,
            title=title,
        )

        image_buffer = io.BytesIO()
        figure.savefig(image_buffer, format="png", dpi=140, bbox_inches="tight")
        plt.close(figure)

        self.charts.append(
            {
                "title": metadata["title"],
                "png_bytes": image_buffer.getvalue(),
                "metadata": metadata,
            }
        )
        return self._record("create_chart", metadata)

    def hypothesis_test(
        self,
        test_type: str,
        numeric_column: str | None = None,
        group_column: str | None = None,
        group_a: str | None = None,
        group_b: str | None = None,
        column_x: str | None = None,
        column_y: str | None = None,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        result = run_hypothesis_test(
            self.dataframe,
            test_type=test_type,
            numeric_column=numeric_column,
            group_column=group_column,
            group_a=group_a,
            group_b=group_b,
            column_x=column_x,
            column_y=column_y,
            alpha=alpha,
        )
        return self._record("hypothesis_test", result)

    def create_feature(
        self,
        operation: str,
        feature1: str,
        new_name: str,
        feature2: str | None = None,
    ) -> dict[str, Any]:
        self.dataframe, result = add_engineered_feature(
            self.dataframe,
            operation=operation,
            feature1=feature1,
            feature2=feature2,
            new_name=new_name,
        )
        return self._record("create_feature", result)

    def train_model(
        self,
        target: str,
        task_type: str = "auto",
        features: list[str] | None = None,
        test_size: float = 0.20,
    ) -> dict[str, Any]:
        result = train_baseline_model(
            self.dataframe,
            target=target,
            task_type=task_type,
            features=features,
            test_size=test_size,
        )
        self.last_model_result = result
        return self._record("train_model", result)

    def _record(self, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        """Store a tool result and return it unchanged."""
        self.activity_log.append({"tool": tool_name, "result": result})
        return result
