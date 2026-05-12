"""Streamlit shell — market risk analytics (overview + tail risk)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

from risk_intel.analytics.metrics import compute_core_metrics_table
from risk_intel.analytics.returns import simple_returns
from risk_intel.analytics.tail_risk import compute_tail_risk_table, qq_plot_points
from risk_intel.config import DEFAULT_TICKERS, RISK_FREE_ANNUAL_DEFAULT
from risk_intel.data.pipeline import load_or_download


st.set_page_config(
    page_title="Global Market Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Global Market Risk Intelligence Platform")
st.caption(
    "Phase 1–2A — live data, core metrics, **tail risk** (VaR/CVaR, skew/kurtosis, normality tests). "
    "Next: correlation, regimes, stress tests, optimisation, AI."
)

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

tab_overview, tab_tail = st.tabs(["Overview", "Tail risk"])

with tab_overview:
    metrics = compute_core_metrics_table(panel, risk_free_annual=rf)
    st.subheader("Core metrics")
    st.dataframe(metrics.round(4), use_container_width=True)

    norm_px = panel.div(
        panel.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else float("nan")),
        axis=1,
    ) * 100.0
    fig = px.line(
        norm_px,
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
        key="dl_prices",
    )

with tab_tail:
    st.subheader("Tail risk and distribution")
    conf = st.select_slider(
        "Confidence level",
        options=[0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x * 100)}%",
    )
    tail_tbl = compute_tail_risk_table(panel, confidence=float(conf))
    st.dataframe(tail_tbl.round(5), use_container_width=True)
    st.caption(
        "VaR and CVaR are **positive loss magnitudes** on **one-period simple returns**. "
        "Historical = empirical quantile / tail mean. Gaussian = normal with sample μ, σ. "
        "Annualised columns use **√252 scaling** (IID shortcut; not a multi-day copula simulation)."
    )

    plot_col = st.selectbox("Ticker for distribution plots", options=list(panel.columns))
    r = simple_returns(panel[plot_col]).dropna()
    if len(r) < 30:
        st.warning("Not enough return observations for stable tail plots.")
    else:
        mu_hat, sig_hat = float(r.mean()), float(r.std(ddof=1))
        xs = np.linspace(float(r.min()), float(r.max()), 200)

        fig_h = go.Figure()
        fig_h.add_trace(
            go.Histogram(
                x=r,
                nbinsx=55,
                name="Empirical",
                histnorm="probability density",
                opacity=0.65,
            )
        )
        fig_h.add_trace(
            go.Scatter(
                x=xs,
                y=norm.pdf(xs, mu_hat, sig_hat),
                name="Normal (sample μ, σ)",
                mode="lines",
            )
        )
        fig_h.update_layout(
            title=f"Return distribution vs normal — {plot_col}",
            xaxis_title="Simple return",
            yaxis_title="Density",
            bargap=0.05,
        )
        st.plotly_chart(fig_h, use_container_width=True)

        tq, sq = qq_plot_points(r)
        if tq.size > 0:
            fig_q = go.Figure()
            fig_q.add_trace(
                go.Scatter(
                    x=tq,
                    y=sq,
                    mode="markers",
                    name="Sample quantiles",
                    marker=dict(size=5, opacity=0.35),
                )
            )
            lim = float(np.nanmax(np.abs(np.concatenate([tq, sq]))))
            fig_q.add_trace(
                go.Scatter(
                    x=[-lim, lim],
                    y=[-lim, lim],
                    mode="lines",
                    name="y = x (Gaussian reference)",
                    line=dict(dash="dash"),
                )
            )
            fig_q.update_layout(
                title=f"Normal Q-Q — {plot_col}",
                xaxis_title="Theoretical quantiles (normal)",
                yaxis_title="Ordered sample returns",
            )
            st.plotly_chart(fig_q, use_container_width=True)

    tbuf = io.StringIO()
    tail_tbl.to_csv(tbuf)
    st.download_button(
        "Download tail risk table CSV",
        data=tbuf.getvalue(),
        file_name="gmri_tail_risk.csv",
        mime="text/csv",
        key="dl_tail",
    )
