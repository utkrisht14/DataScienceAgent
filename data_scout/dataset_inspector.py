"""
Dataset inspection and compactcontext creation.

Every function in this file accepts a normal pandas DataFrame.
It doesn't require the AnalysisEngine, which keeps the dependency direction simple.
"""

from __future__ import annotations

import json

from typing import Any

import pandas as pd

from data_scout.serialization import dataframe_preview, to_json_safe

def inspect_dataset(dataframe: pd.DataFrame) ->dict[str, Any]:
    """ Return shape, columns, types, missing values, and df.head() """
    return {
        "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "columns": list(dataframe.columns),
        "data_types": {
            column: str(dtype) for column, dtype in dataframe.dtypes.items()
        },
        "missing_values": {
            column: int(value)
            for column, value in dataframe.isna().sum().items()
        },
        "unique_values": {
            column: int(dataframe[column].nunique(dropna=True))
            for column in dataframe.columns
        },
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "head": dataframe_preview(dataframe, rows=5),
    }


def build_agent_context(dataframe: pd.DataFrame)-> str:
    """
    Create a compact dataset summary for the OpenAI system instructions.

    The complete CSV is not sent to the model. Only the schema, missing-value
    counts, unique counts, and three sample rows are included.
    """

    context = {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "data_types":{
            column: str(dtype) for column, dtype in dataframe.dtypes.items()
        },
        "missing_values": {
            column: int(value)
            for column, value in dataframe.isna().sum().items()
        },
        "unique_values": {
            column: int(dataframe[column].nunique(dropna=True))
            for column in dataframe.columns
        },
        "sample_rows": dataframe_preview(dataframe, rows=3),
    }
    return json.dumps(to_json_safe(context), ensure_ascii=False)