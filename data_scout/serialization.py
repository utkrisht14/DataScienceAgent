"""Shared helper functions for Data Scout."""

from __future__ import annotations

import json
import pandas as pd
import numpy as np

from typing import Any


def to_json_safe(value: Any) -> Any:
    """Recursively convert a value into a JSON-compatible Python object."""
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None

    # pd.isna works for scalar values, but not safely for lists or dictionaries.
    if not isinstance(value, (list, tuple, dict, set)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

    return value

def dataframe_preview(dataframe: pd.DataFrame, rows:int=5) -> str:
    """ Returns a small DataFrame preview as JSON-safe records. """
    records = dataframe.head(rows).to_dict(orient="records")
    return to_json_safe(records)


def to_json_text(value: Any) ->str:
    """ Convert any value into a formatted JSON text. """
    return json.dumps(to_json_safe(value), ensure_ascii=False, indent=2)
