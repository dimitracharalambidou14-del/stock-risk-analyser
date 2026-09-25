# data.py
# Everything to do with downloading market data lives in this file.
# app.py handles what the page looks like; this file handles getting the numbers.

import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Settings for the time-range buttons.
# Names written in CAPITALS are "constants": values we set once and never change.
# ---------------------------------------------------------------------------

# Short ranges need INTRADAY data (several prices per day).
# Each entry is a pair: (how far back, how often).
INTRADAY_SETTINGS = {
    "1D": ("1d", "5m"),   # the latest trading day, one price every 5 minutes
    "1W": ("5d", "30m"),  # the last 5 trading days, one price every 30 minutes
}

# Longer ranges are cut from the 5 years of DAILY data we already download.
DAILY_LOOKBACK = {
    "1M": pd.DateOffset(months=1),
    "1Y": pd.DateOffset(years=1),
    "5Y": pd.DateOffset(years=5),
}


# "@st.cache_data" is a DECORATOR: a label that adds extra behaviour to the
# function underneath it. Here it tells Streamlit to remember (cache) the result,
# so the same data isn't downloaded again on every rerun.
# ttl = "time to live": how many seconds to keep the saved copy.
# If the function crashes with an error, nothing is cached, so a
# temporary internet problem won't get "remembered".
@st.cache_data(ttl=86400, show_spinner=False)
def search_tickers(query: str) -> dict[str, str]:
    """Search Yahoo Finance for shares or funds matching the text typed.

    Returns a DICTIONARY (a lookup table) where each key is a readable label
    and each value is the ticker, for example:
        {"Apple Inc. (AAPL, NASDAQ)": "AAPL"}
    Returns an empty dictionary if nothing matches.
    Results are kept for 86400 seconds (1 day): company names rarely change.
    """
    # Ask Yahoo for up to 10 matches. ".quotes" is a LIST of results.
    results = yf.Search(query, max_results=10).quotes

    matches = {}  # start with an empty lookup table

    # A FOR LOOP repeats the indented lines once for each item in the list.
    for item in results:
        # Keep only company shares (EQUITY) and funds (ETF).
        if item.get("quoteType") in ("EQUITY", "ETF"):
            symbol = item.get("symbol")
            # "or" picks the first one that isn't empty: long name, else short name.
            name = item.get("longname") or item.get("shortname") or symbol
            exchange = item.get("exchDisp") or item.get("exchange") or "unknown exchange"

            label = f"{name} ({symbol}, {exchange})"
            matches[label] = symbol  # add a row to the lookup table

    return matches


@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download DAILY price history for one ticker from Yahoo Finance.

    ticker: the stock symbol, for example "AAPL" or "HSBA.L".
    period: how far back to go, for example "1y" or "5y".

    Returns a table (a pandas DataFrame) with one row per trading day.
    If the ticker doesn't exist, the table comes back empty.
    Results are kept for 3600 seconds (1 hour).
    """
    # auto_adjust=True adjusts past prices for dividends and stock splits,
    # so returns calculated from these prices are TOTAL returns.
    history = yf.Ticker(ticker).history(period=period, auto_adjust=True)

    if not history.empty:
        # Remove time-zone information from the dates, so dates from
        # different exchanges can line up when we compare stocks.
        history.index = history.index.tz_localize(None)

    return history


@st.cache_data(ttl=300, show_spinner=False)
def get_intraday_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download INTRADAY prices (several per day) for one ticker.

    period: how far back to go, for example "1d" or "5d".
    interval: how often, for example "5m" (every 5 minutes) or "30m".

    Results are kept for only 300 seconds (5 minutes),
    because intraday prices change quickly.
    """
    history = yf.Ticker(ticker).history(
        period=period, interval=interval, auto_adjust=True
    )

    if not history.empty:
        # Remove time-zone information, keeping the exchange's local clock time.
        history.index = history.index.tz_localize(None)

    return history


def get_chart_data(ticker: str, time_range: str) -> pd.DataFrame:
    """Return the prices to draw for the chosen time-range button.

    time_range: one of "1D", "1W", "1M", "1Y", "5Y".
    """
    if time_range in INTRADAY_SETTINGS:
        # "Unpack" the pair: ("1d", "5m") becomes period = "1d", interval = "5m".
        period, interval = INTRADAY_SETTINGS[time_range]
        return get_intraday_history(ticker, period, interval)

    daily = get_price_history(ticker)
    if daily.empty:
        return daily

    # Start date = the most recent date minus the look-back period.
    start_date = daily.index[-1] - DAILY_LOOKBACK[time_range]

    # Keep only the rows on or after the start date (like a filter in Excel).
    return daily[daily.index >= start_date]


def get_aligned_closes(tickers: list[str], time_range: str = "1Y") -> pd.DataFrame:
    """Build ONE table of daily closing prices for several tickers, lined up by date.

    tickers: a list of stock symbols, for example ["AAPL", "TSCO.L"].
    time_range: "1M", "1Y" or "5Y".

    Each column of the result is one ticker; each row is one date.
    Tickers with no data are left out.
    """
    columns = {}  # a dictionary: ticker -> its column of closing prices

    for ticker in tickers:
        history = get_price_history(ticker)
        if not history.empty:
            columns[ticker] = history["Close"]

    if not columns:
        # Nothing downloaded successfully: return an empty table.
        return pd.DataFrame()

    # Turning the dictionary into a DataFrame lines the columns up by date.
    # On dates when one exchange was closed (e.g. a UK bank holiday but not a
    # US one), that stock's cell is blank - shown as NaN ("not a number").
    closes = pd.DataFrame(columns)

    # Fill each blank with the previous day's price ("forward fill").
    # Reasoning: if an exchange was closed, its price simply didn't change.
    closes = closes.ffill()

    # Remove rows that are still blank: dates before every stock has a price
    # (for example, if one company only listed recently).
    closes = closes.dropna()

    if closes.empty:
        return closes

    start_date = closes.index[-1] - DAILY_LOOKBACK[time_range]
    return closes[closes.index >= start_date]