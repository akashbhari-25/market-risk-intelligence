"""Streamlit shell — market risk analytics (overview + tail risk)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

from risk_intel.analytics.correlation import (
    CRISIS_WINDOWS,
    full_sample_correlation,
    return_panel,
    rolling_average_pairwise_correlation,
    rolling_pair_correlation,
    upper_triangle_mean,
)
from risk_intel.analytics.metrics import compute_core_metrics_table
from risk_intel.analytics.regime import (
    drawdown_series,
    market_regime_from_drawdown,
    rolling_beta,
    rolling_sharpe,
    rolling_volatility,
)
from risk_intel.analytics.returns import simple_returns
from risk_intel.analytics.stress import stress_scenario_names, stress_table_for_scenario
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
    "Phase 1–4 — live data, core metrics, **tail risk**, **correlation**, **regime/rolling risk**, "
    "and **stress scenarios**. Next: portfolio optimisation, AI."
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

tab_overview, tab_tail, tab_corr, tab_regime, tab_stress = st.tabs(
    ["Overview", "Tail risk", "Correlation", "Regime & Rolling", "Stress"]
)

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

with tab_corr:
    st.subheader("Correlation and diversification context")
    rets = return_panel(panel)
    valid_rows = int(rets.dropna(how="all").shape[0])
    if valid_rows < 20:
        st.warning("Need more overlapping return observations for stable correlations.")
    else:
        corr_full = full_sample_correlation(rets)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Avg pairwise correlation (full sample)", f"{upper_triangle_mean(corr_full):.3f}")
        with c2:
            st.caption("Pearson on **simple returns**, pairwise-complete rows.")

        fig_full = px.imshow(
            corr_full,
            text_auto=".2f",
            aspect="equal",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Full-sample correlation matrix",
        )
        fig_full.update_layout(xaxis_side="top")
        st.plotly_chart(fig_full, use_container_width=True)

        st.markdown("#### Crisis-window correlations")
        crisis_name = st.selectbox("Preset window", options=list(CRISIS_WINDOWS.keys()))
        t0, t1 = CRISIS_WINDOWS[crisis_name]
        crisis_rets = rets.loc[(rets.index >= pd.Timestamp(t0)) & (rets.index <= pd.Timestamp(t1))]
        n_c = int(crisis_rets.dropna(how="all").shape[0])
        if n_c < 10:
            st.warning(
                f"Not enough dates in **{crisis_name}** for this sample/frequency "
                f"(rows in slice ≈ {n_c}). Try daily data or a wider overall range."
            )
        else:
            min_p = max(5, min(10, n_c // 3))
            corr_crisis = full_sample_correlation(crisis_rets, min_periods=min_p)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.metric("Avg pairwise (crisis window)", f"{upper_triangle_mean(corr_crisis):.3f}")
            with cc2:
                st.metric("Δ vs full sample", f"{upper_triangle_mean(corr_crisis) - upper_triangle_mean(corr_full):.3f}")
            fig_c = px.imshow(
                corr_crisis,
                text_auto=".2f",
                aspect="equal",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title=f"Correlation — {crisis_name}",
            )
            fig_c.update_layout(xaxis_side="top")
            st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("#### Rolling correlations")
        rwin = st.select_slider("Rolling window (sessions)", options=[63, 126, 252], value=126)
        cols_list = list(panel.columns)
        if len(cols_list) >= 2:
            r1, r2 = st.columns(2)
            with r1:
                a = st.selectbox("Series A", options=cols_list, index=0, key="roll_a")
            with r2:
                b = st.selectbox("Series B", options=cols_list, index=min(1, len(cols_list) - 1), key="roll_b")
            if a == b:
                st.info("Pick two different tickers for the pair chart.")
            else:
                roll_ab = rolling_pair_correlation(rets, a, b, int(rwin)).dropna()
                fig_ab = px.line(
                    roll_ab,
                    labels={"value": "Correlation", "index": "Date"},
                    title=f"Rolling {rwin}-period correlation: {a} vs {b}",
                )
                fig_ab.update_layout(showlegend=False)
                st.plotly_chart(fig_ab, use_container_width=True)

            roll_avg = rolling_average_pairwise_correlation(rets, int(rwin)).dropna()
            fig_avg = px.line(
                roll_avg,
                labels={"value": "Mean pairwise ρ", "index": "Date"},
                title=f"Rolling average pairwise correlation ({rwin}-period)",
            )
            fig_avg.update_layout(showlegend=False)
            st.plotly_chart(fig_avg, use_container_width=True)
        else:
            st.info("Select at least two tickers in the sidebar for rolling correlation charts.")

        cbuf = io.StringIO()
        corr_full.to_csv(cbuf)
        st.download_button(
            "Download full-sample correlation CSV",
            data=cbuf.getvalue(),
            file_name="gmri_correlation_matrix.csv",
            mime="text/csv",
            key="dl_corr",
        )

with tab_regime:
    st.subheader("Regime and rolling risk")
    rets = simple_returns(panel)
    tickers_list = list(panel.columns)

    c1, c2 = st.columns(2)
    with c1:
        sel_ticker = st.selectbox("Ticker", options=tickers_list, key="reg_ticker")
    with c2:
        default_bm_idx = tickers_list.index("^GSPC") if "^GSPC" in tickers_list else 0
        sel_benchmark = st.selectbox(
            "Benchmark",
            options=tickers_list,
            index=default_bm_idx,
            key="reg_benchmark",
        )

    k1, k2 = st.columns(2)
    with k1:
        rwin = st.select_slider("Rolling window (sessions)", options=[63, 126, 252], value=126, key="reg_window")
    with k2:
        bear_threshold = st.number_input(
            "Bear threshold (drawdown)",
            min_value=-0.80,
            max_value=-0.05,
            value=-0.20,
            step=0.01,
            format="%.2f",
            key="reg_bear_threshold",
        )

    px_sel = panel[sel_ticker]
    r_sel = rets[sel_ticker]
    r_bm = rets[sel_benchmark]

    roll_vol = rolling_volatility(r_sel, window=int(rwin)).dropna()
    roll_sh = rolling_sharpe(r_sel, rf_annual=float(rf), window=int(rwin)).dropna()
    roll_beta = rolling_beta(r_sel, r_bm, window=int(rwin)).dropna()
    dd = drawdown_series(px_sel)
    regime = market_regime_from_drawdown(dd, bear_threshold=float(bear_threshold))

    if not roll_vol.empty:
        fig_vol = px.line(
            roll_vol,
            labels={"value": "Annualised volatility", "index": "Date"},
            title=f"Rolling {rwin}-period volatility — {sel_ticker}",
        )
        fig_vol.update_layout(showlegend=False)
        st.plotly_chart(fig_vol, use_container_width=True)

    if not roll_sh.empty:
        fig_sh = px.line(
            roll_sh,
            labels={"value": "Sharpe", "index": "Date"},
            title=f"Rolling {rwin}-period Sharpe — {sel_ticker}",
        )
        fig_sh.update_layout(showlegend=False)
        st.plotly_chart(fig_sh, use_container_width=True)

    if not roll_beta.empty:
        fig_beta = px.line(
            roll_beta,
            labels={"value": "Beta", "index": "Date"},
            title=f"Rolling {rwin}-period beta: {sel_ticker} vs {sel_benchmark}",
        )
        fig_beta.update_layout(showlegend=False)
        st.plotly_chart(fig_beta, use_container_width=True)

    if not dd.empty:
        fig_dd = px.line(
            dd,
            labels={"value": "Drawdown", "index": "Date"},
            title=f"Drawdown path — {sel_ticker}",
        )
        fig_dd.add_hline(y=float(bear_threshold), line_dash="dash", line_color="red")
        fig_dd.update_layout(showlegend=False)
        st.plotly_chart(fig_dd, use_container_width=True)

    if regime.empty:
        st.info("Not enough data to label regimes.")
    else:
        pct_bull = float((regime == "Bull").mean() * 100.0)
        pct_bear = float((regime == "Bear").mean() * 100.0)
        current_regime = regime.iloc[-1]
        current_dd = float(dd.dropna().iloc[-1]) if not dd.dropna().empty else float("nan")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("% time Bull", f"{pct_bull:.1f}%")
        s2.metric("% time Bear", f"{pct_bear:.1f}%")
        s3.metric("Current regime", str(current_regime))
        s4.metric("Current drawdown", f"{current_dd:.2%}")

with tab_stress:
    st.subheader("Stress testing (historical windows)")
    interval_ctx = str(st.session_state.get("interval", interval))
    st.caption(
        "Metrics use **simple returns** on your selected sampling frequency. "
        "Annualised vol / Sharpe scale with **252 (daily)**, **52 (weekly)**, or **12 (monthly)**. "
        "Equal-weight row is a **1/N rebalanced each period** proxy (not transaction-cost adjusted)."
    )
    scenario = st.selectbox("Scenario", options=stress_scenario_names(), index=0, key="stress_scenario")
    stress_tbl = stress_table_for_scenario(panel, scenario, interval_ctx, float(rf))
    if stress_tbl.empty:
        st.warning("No overlapping data in this window for the current fetch. Widen the date range or use daily data.")
    else:
        st.dataframe(stress_tbl.round(4), use_container_width=True)
        sbuf = io.StringIO()
        stress_tbl.to_csv(sbuf)
        st.download_button(
            "Download stress table CSV",
            data=sbuf.getvalue(),
            file_name="gmri_stress.csv",
            mime="text/csv",
            key="dl_stress",
        )
