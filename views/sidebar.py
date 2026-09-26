# views/sidebar.py
# The left-hand panel: settings shared by every tab (history window, risk-free rate,
# inflation), plus "About".

import pandas as pd
import streamlit as st

from data import get_uk_bank_rate_history, get_uk_cpi_inflation, get_us_treasury_bill_history
from views.common import FALLBACK_INFLATION, FALLBACK_RISK_FREE, INFLATION_SOURCES, RISK_FREE_SOURCES


def choose_risk_free_rate() -> float | pd.Series:
    """Sidebar control: where the risk-free rate comes from.

    Returns either a Series of DAILY rates (for the live sources), so each calculation
    can use the rates that actually applied during its own period, or one fixed rate
    (for Custom, or if the download fails).
    """
    source = st.selectbox(
        "Risk-free rate",
        RISK_FREE_SOURCES,
        key="rf_source",
        help=(
            "The return available with (almost) no risk, used in Sharpe and Sortino ratios. "
            "With a live source, each calculation uses the average rate over its own period "
            "(e.g. 2021-2026), not just today's rate. UK Bank Rate suits a UK investor; the "
            "US Treasury bill rate suits US stocks."
        ),
    )

    if source == "Custom":
        custom_pct = st.number_input(
            "Your risk-free rate (% per year)",
            min_value=0.0, max_value=20.0, value=4.0, step=0.25, key="risk_free_pct",
        )
        st.caption("A fixed rate, used for every period.")
        return custom_pct / 100

    try:
        if source == "UK Bank Rate (latest)":
            history = get_uk_bank_rate_history()
            origin = "Bank of England"
        else:
            history = get_us_treasury_bill_history()
            origin = "US Treasury, via Yahoo Finance"
        st.caption(
            f"Today: **{history.iloc[-1]:.2%}** on {history.index[-1]:%d %b %Y} ({origin}). "
            "Past periods use the rates that applied at the time."
        )
        return history
    except Exception:
        st.warning(
            f"Couldn't download the latest rates right now, so a fixed {FALLBACK_RISK_FREE:.2%} "
            "is used. Choose Custom to set your own."
        )
        return FALLBACK_RISK_FREE


def choose_inflation() -> float:
    """Sidebar control: where the inflation assumption comes from. Returns it as a decimal."""
    source = st.selectbox(
        "Inflation",
        INFLATION_SOURCES,
        key="inflation_source",
        help=(
            "Used by the goal planner to show amounts in today's money. The latest figure "
            "describes the past 12 months and may not last; for long-term plans, the Bank "
            "of England's 2% target is a common assumption."
        ),
    )

    if source == "Custom":
        custom_pct = st.number_input(
            "Your inflation assumption (% per year)",
            min_value=0.0, max_value=15.0, value=2.0, step=0.25, key="inflation_pct",
        )
        return custom_pct / 100

    if source == "Bank of England 2% target":
        st.caption("The Bank of England's target for CPI inflation.")
        return 0.02

    try:
        rate, month = get_uk_cpi_inflation()
        st.caption(f"**{rate:.1%}** over the 12 months to {month:%B %Y} (Office for National Statistics).")
        return rate
    except Exception:
        st.warning(
            f"Couldn't download the latest inflation figure right now, so {FALLBACK_INFLATION:.1%} "
            "is used. Choose Custom to set your own."
        )
        return FALLBACK_INFLATION


def show_sidebar() -> tuple[str, float | pd.Series, float]:
    """Draw the sidebar and return the settings chosen there:
    (history window label, risk-free rate or rate history, inflation)."""
    # Everything inside "with st.sidebar:" appears in the panel on the left.
    with st.sidebar:
        st.header("Settings")
        st.caption("These apply to the risk analysis and the portfolio tools.")

        window_label = st.segmented_control(
            "History used for the estimates",
            options=["1Y", "5Y"],
            default="5Y",
            key="history_window",
            help=(
                "How many years of past daily prices the risk measures and portfolio "
                "estimates use. More history gives steadier estimates; less history "
                "reflects recent conditions more closely."
            ),
        )

        st.subheader("Economic assumptions")
        st.caption("Live figures from official sources, or choose your own.")
        risk_free = choose_risk_free_rate()
        inflation = choose_inflation()

        st.divider()
        st.header("About")
        st.write(
            "An educational project that measures the past risk of shares and illustrates "
            "portfolio theory, built with Python, Streamlit, pandas, NumPy, SciPy and Plotly."
        )
        st.caption(
            "Data: Yahoo Finance, via the free yfinance library. Prices may be delayed "
            "and can occasionally contain errors."
        )

    if window_label is None:
        window_label = "5Y"

    return window_label, risk_free, inflation
