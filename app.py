"""Streamlit shell — Phase 1: data pull, core metrics, normalised performance."""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from risk_intel.analytics.metrics import compute_core_metrics_table
from risk_intel.config import DEFAULT_TICKERS, RISK_FREE_ANNUAL_DEFAULT
from risk_intel.data.pipeline import load_or_download


st.set_page_config(
    page_title="Global Market Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Global Market Risk Intelligence Platform")
st.caption("Phase 1 — live data, caching, institutional core metrics. Later phases add tail risk, regimes, stress tests, optimisers, AI.")

with st.sidebar:
    st.header("Controls")
    tickers = st.multiselect(
        "Indices / tickers",
        options=list(DEFAULT_TICKERS),
        default=list(DEFAULT_TICKERS)[:5],
    )
    col_a, col_b = st.columns(2)
    with col_a:
        start = st.date_input("Start", value=pd.Timestamp("2015-01-01").date())
    with col_b:
        end = st.date_input("End", value=pd.Timestamp.today().date())
    interval = st.selectbox("Sampling", options=["1d", "1wk", "1mo"], index=0)
    rf = st.number_input(
        "Annual risk-free rate",
        min_value=0.0,
        max_value=0.2,
        value=float(RISK_FREE_ANNUAL_DEFAULT),
        step=0.005,
        format="%.4f",
    )
    force_refresh = st.checkbox("Force refresh (bypass cache)", value=False)
    run = st.button("Run analysis", type="primary")

if not tickers:
    st.warning("Select at least one ticker.")
    st.stop()

if not run and "panel" not in st.session_state:
    st.info("Configure the sidebar and press **Run analysis**.")
    st.stop()

if run:
    with st.spinner("Fetching and validating market data…"):
        panel, manifest = load_or_download(
            list(tickers),
            start.isoformat(),
            end.isoformat(),
            interval=interval,  # type: ignore[arg-type]
            force_refresh=force_refresh,
        )
    st.session_state["panel"] = panel
    st.session_state["manifest"] = manifest
    st.session_state["interval"] = interval

panel: pd.DataFrame = st.session_state["panel"]
manifest = st.session_state["manifest"]

if panel.empty:
    st.error("No price data returned. Try a wider date range, fewer tickers, or force refresh.")
    st.json({"warnings": manifest.warnings, "cache_key": manifest.cache_key})
    st.stop()

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Rows (dates)", f"{manifest.rows:,}")
mcol2.metric("Tickers loaded", len(panel.columns))
mcol3.metric("Cache", "hit" if manifest.from_cache else "miss / refreshed")

if manifest.warnings:
    with st.expander("Data quality warnings", expanded=False):
        for w in manifest.warnings:
            st.write(f"- {w}")

metrics = compute_core_metrics_table(panel, risk_free_annual=rf)
st.subheader("Core metrics")
st.dataframe(metrics.round(4), use_container_width=True)

norm = panel.div(panel.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else float("nan")), axis=1) * 100.0
fig = px.line(
    norm,
    labels={"value": "Index (normalised to 100)", "index": "Date", "variable": "Ticker"},
    title="Normalised total-return levels (adjusted close)",
)
fig.update_layout(legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

buf = io.BytesIO()
panel.to_csv(buf)
st.download_button(
    "Download prices CSV",
    data=buf.getvalue(),
    file_name="gmri_prices.csv",
    mime="text/csv",
)
