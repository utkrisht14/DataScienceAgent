"""
CSV file loading.

This is the real data-acquisition module. It knows how to read CSV input, but
it does not know anything about the AnalysisEngine, Streamlit, or OpenAI.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd

CsvSource =str | Path |BinaryIO |io.BytesIO

def load_csv(source: CsvSource)-> pd.DataFrame:
    """
    Load a CSV source into the pandas DataFrame.

    Args:
        source: A filepath or a binary file-like object, including a Streamlit UploadedFilewrapped in a BytesIO object.

    Raises:
        ValueError: When the file is empty, malformed, or contains no rows.
    """

    try:
        dataframe = pd.read_csv(source)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The CSV file is empty.") from error
    except pd.errors.ParserError as error:
        raise ValueError("The CSV file could not be parsed.") from error
    except UnicodeDecodeError as error:
        raise ValueError("The CSV file is not UTF-8 encoded.") from error
    except Exception as error:
        raise ValueError(f"Could not load the CSV file: {error}") from error

    if dataframe.empty:
        raise ValueError("The CSV file contains no rows.")

    return dataframe


def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Convenience wrapper used by the Streamlit interface."""
    return load_csv(io.BytesIO(file_bytes))




