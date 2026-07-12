"""Shared helper functions for Data Scout."""

from __future__ import annotations

import io
import pandas as pd
import numpy as np

from typing import Any


def json_safe(value: Any) -> Any:
    """Convert NumPy and pandas values into JSON-compatible values."""

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, float) and (
        np.isnan(value) or np.isinf(value)
    ):
        return None

    if pd.isna(value):
        return None

    return value

def dataframe_records(frame: pd.DataFrame, max_rows: int=20):
    """Return a small JSON-safe preview of a DataFrame."""
    return json_safe(frame.head(max_rows).to_dict(orient='records'))

def read_upload_csv(file_bytes: bytes)-> pd.DataFrame:
    """Read a CSV file with common encoding fallbacks."""
    last_error: Exception | None = None

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding) # Creates in-memory binary files
        except UnicodeError as exc:
            last_error = exc

    raise ValueError(f"Failed to read CSV file: {last_error}")

