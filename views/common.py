# views/common.py
# Settings and small tools shared by several parts of the page.
# Keeping them in one place means each setting is defined once (a single source of truth).

import pandas as pd
import streamlit as st

# Market indices available for beta. On Yahoo Finance, "^" marks an index.
BENCHMARKS = {
    "S&P 500 (US)": "^GSPC",
    "FTSE 100 (UK)": "^FTSE",
    "Nasdaq 100 (US)": "^NDX",
    "Euro Stoxx 50 (Europe)": "^STOXX50E",
}

# Confidence levels for VaR and Expected Shortfall.
CONFIDENCE_LEVELS = {"95%": 0.95, "99%": 0.99}

# Where the economic assumptions can come from (chosen in the sidebar).
RISK_FREE_SOURCES = ["UK Bank Rate (latest)", "US 3-month Treasury bill (latest)", "Custom"]
INFLATION_SOURCES = ["UK CPI inflation (latest)", "Bank of England 2% target", "Custom"]

# Used only if the live data can't be downloaded.
FALLBACK_RISK_FREE = 0.04
FALLBACK_INFLATION = 0.02

# The one-click example portfolio: two US shares and one UK share.
EXAMPLE_TICKERS = ["AAPL", "MSFT", "TSCO.L"]

# Return assumptions offered by the goal planner.
RETURN_BASES = ["Long-run (cautious)", "Historical", "Custom"]

# Settings for the strategy test: button label -> number.
LOOKBACK_YEARS = {"1Y": 1, "2Y": 2, "3Y": 3}
REBALANCE_MONTHS = {"Every 6 months": 6, "Every year": 12}

# The disclaimer shown at the top of every page.
DISCLAIMER = (
    "**Educational tool, not financial advice.** This app describes how shares behaved "
    "in the past and illustrates portfolio theory. It never recommends buying or selling "
    "anything, and past performance does not predict future results."
)


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------


def format_return(value: float | None) -> str:
    """Turn a return like 0.123 into text like "+12.3%" ("n/a" if missing)."""
    if value is None:
        return "n/a"
    if abs(value) >= 1:
        # 100% or more: drop the decimal so the number fits its column.
        return f"{value:+.0%}"
    return f"{value:+.1%}"


def delta_or_none(difference: float, text: str, decimal_places: int) -> str | None:
    """Return the "change" text for st.metric, or None if the change rounds to zero.

    Without this, a tiny difference like +0.0003% would show a coloured arrow
    and "+0.0%", suggesting a change that isn't really there.
    """
    if round(difference, decimal_places) == 0:
        return None  # st.metric shows no arrow when delta is None
    return text


def latest_risk_free_rate(risk_free: float | pd.Series) -> float:
    """Today's risk-free rate: the last value of a rate history, or the fixed rate itself."""
    if isinstance(risk_free, pd.Series):
        return float(risk_free.iloc[-1])
    return float(risk_free)


def load_example_portfolio() -> None:
    """Button "callback": runs BEFORE the page is rebuilt, so it can fill in widgets.

    It puts three example stocks in the comparison list, types "Apple" into the
    search box and opens Apple's risk analysis, so a first-time visitor can see
    every tab working with one click.
    """
    st.session_state.watchlist = list(EXAMPLE_TICKERS)
    st.session_state.search_query = "Apple"
    st.session_state.analysed_ticker = "AAPL"



# The text of the panel at the bottom of the page. It's written in Markdown, where
# "**bold**" makes bold text and lines with "|" make a table. It sits at the far left
# on purpose: in Markdown, lines indented by 4 spaces would turn into a code block.
DATA_SOURCES_TEXT = """
**Live data** (downloaded when you use the app; free, no keys)

| What | Source | How fresh |
|---|---|---|
| Share and index prices | Yahoo Finance, via the yfinance library (adjusted for dividends and splits) | Re-downloaded at most every hour; intraday every 5 minutes, possibly delayed about 15 minutes |
| UK Bank Rate | Bank of England database (series IUDBEDR) | Re-downloaded every 12 hours |
| US 3-month Treasury bill yield | Yahoo Finance (^IRX) | Re-downloaded every 12 hours |
| UK CPI inflation | Office for National Statistics (series D7G7) | Re-downloaded every 12 hours; published monthly |

**Assumptions** (chosen, not data; most can be changed in the app)

| Assumption | Value |
|---|---|
| Trading days per year | 252 |
| Value at Risk and Expected Shortfall | 1-day horizon, 95% or 99% confidence |
| Risk aversion (low / medium / high appetite) | 10 / 4 / 1.5 |
| Monte Carlo | 10,000 simulations, fixed random seed (42) |
| Goal planner return (long-run) | Today's risk-free rate + 0.30 x the portfolio's volatility (can switch to historical or custom) |
| Goal planner defaults | £10,000 start, £200 a month, 0.5% fees, £50,000 goal (all editable) |
| Example portfolio | Apple (AAPL), Microsoft (MSFT), Tesco (TSCO.L): an illustration, not a recommendation |
| Strategy test defaults | 2-year look-back, yearly rebalancing, 0.10% trading cost (all editable) |
| If live rates can't be downloaded | 4% risk-free rate, 2% inflation (clearly flagged) |

**Model outputs** (everything else: returns, risk measures, optimised weights and projections)
are calculated from the data above using the stated methods. They describe the past or
illustrate possibilities; they are not forecasts.
"""


def show_data_sources_and_assumptions() -> None:
    """The panel at the bottom of the page: which numbers are live data, and which are assumptions."""
    with st.expander("Data sources and assumptions"):
        st.markdown(DATA_SOURCES_TEXT)
