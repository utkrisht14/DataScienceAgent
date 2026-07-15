"""Streamlit interface for Data Scout."""

from __future__ import annotations

import hashlib
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from data_scout.agent.openai_agent import run_data_scout_agent
from data_scout.analysis_engine import AnalysisEngine
from data_scout.csv_loader import load_csv_bytes
from data_scout.serialization import to_json_safe

load_dotenv()


def run_app() -> None:
    """Render the complete Streamlit application."""
    _configure_page()
    api_key, model = _render_sidebar()

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="The complete CSV stays in the local Streamlit process.",
    )
    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        return

    engine = _get_or_create_engine(uploaded_file.getvalue())
    if engine is None:
        return

    _show_dataset_preview(engine)
    user_request = _get_analysis_request(engine)
    _handle_buttons(engine, user_request, api_key, model)
    _show_results(engine)


def _configure_page() -> None:
    st.set_page_config(page_title="Data Scout", page_icon="🔎", layout="wide")
    st.title("🔎 Data Scout")
    st.caption(
        "An educational CSV data-science agent using safe local tools and "
        "the OpenAI Responses API."
    )


def _render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.header("Configuration")

        entered_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="You can instead store OPENAI_API_KEY in a local .env file.",
        )
        api_key = entered_key or os.getenv("OPENAI_API_KEY", "")

        model = st.selectbox(
            "Model",
            options=["gpt-4.1-mini", "gpt-4.1"],
            index=0,
            help="gpt-4.1-mini is the lower-cost learning default.",
        )

        st.info(
            "The model receives compact summaries and tool results, not the "
            "complete CSV file."
        )

    return api_key, model


def _get_or_create_engine(file_bytes: bytes) -> AnalysisEngine | None:
    """Keep one engine in Streamlit session state for the uploaded file."""
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.get("data_scout_file_hash") != file_hash:
        try:
            dataframe = load_csv_bytes(file_bytes)
        except ValueError as error:
            st.error(str(error))
            return None

        st.session_state.data_scout_file_hash = file_hash
        st.session_state.data_scout_engine = AnalysisEngine(dataframe)
        st.session_state.data_scout_report = ""

    return st.session_state.data_scout_engine


def _show_dataset_preview(engine: AnalysisEngine) -> None:
    st.subheader("1. Data acquisition")
    st.write(
        f"Loaded **{len(engine.dataframe):,} rows** and "
        f"**{engine.dataframe.shape[1]} columns**."
    )
    st.dataframe(engine.dataframe.head(), use_container_width=True)

    type_column, missing_column = st.columns(2)

    with type_column:
        st.markdown("**Data types**")
        type_frame = pd.DataFrame(
            {
                "column": engine.dataframe.columns,
                "dtype": engine.dataframe.dtypes.astype(str).values,
            }
        )
        st.dataframe(type_frame, hide_index=True, use_container_width=True)

    with missing_column:
        st.markdown("**Missing values**")
        missing_frame = (
            engine.dataframe.isna()
            .sum()
            .rename("missing")
            .reset_index(names="column")
        )
        st.dataframe(missing_frame, hide_index=True, use_container_width=True)


def _get_analysis_request(engine: AnalysisEngine) -> str:
    st.subheader("2. Ask Data Scout")

    target_options = ["No predictive model"] + list(engine.dataframe.columns)
    target = st.selectbox(
        "Optional predictive-model target",
        options=target_options,
    )

    default_request = (
        "Inspect and clean the dataset, summarize its most important patterns, "
        "create two useful visualizations, and run one defensible hypothesis "
        "test. Finish with practical conclusions for a non-technical reader."
    )
    request = st.text_area(
        "Analysis request",
        value=default_request,
        height=140,
    )

    if target == "No predictive model":
        request += "\nDo not train a predictive model."
    else:
        request += f"\nTrain a baseline predictive model with target: {target}"

    return request


def _handle_buttons(
    engine: AnalysisEngine,
    user_request: str,
    api_key: str,
    model: str,
) -> None:
    run_column, reset_column = st.columns(2)

    with run_column:
        run_clicked = st.button(
            "Run Data Scout",
            type="primary",
            use_container_width=True,
        )

    with reset_column:
        reset_clicked = st.button(
            "Reset working data",
            use_container_width=True,
        )

    if reset_clicked:
        engine.reset()
        st.session_state.data_scout_report = ""
        st.rerun()

    if not run_clicked:
        return

    if not api_key:
        st.error("Add OPENAI_API_KEY to .env or enter it in the sidebar.")
        return

    engine.charts.clear()

    try:
        with st.spinner("Data Scout is using local analysis tools..."):
            report = run_data_scout_agent(
                engine=engine,
                user_request=user_request,
                api_key=api_key,
                model=model,
            )
        st.session_state.data_scout_report = report
    except Exception as error:
        st.error(f"The agent failed: {type(error).__name__}: {error}")


def _show_results(engine: AnalysisEngine) -> None:
    report = st.session_state.get("data_scout_report", "")
    if not report:
        return

    st.subheader("3. Data Scout report")
    st.markdown(report)

    if engine.charts:
        st.subheader("Generated charts")
        for chart in engine.charts:
            st.image(
                chart["png_bytes"],
                caption=chart["title"],
                use_container_width=True,
            )

    st.subheader("Downloads")
    csv_column, report_column = st.columns(2)

    with csv_column:
        st.download_button(
            "Download working CSV",
            data=engine.dataframe.to_csv(index=False).encode("utf-8"),
            file_name="data_scout_working_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with report_column:
        st.download_button(
            "Download report",
            data=report.encode("utf-8"),
            file_name="data_scout_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with st.expander("Tool activity log"):
        st.json(to_json_safe(engine.activity_log))
