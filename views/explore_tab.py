# views/explore_tab.py
# Tab 1, "Explore a stock": search for a stock, chart its price, and analyse its risk.

import numpy as np
import pandas as pd
import streamlit as st

from backtest import average_risk_free_rate
from charts import beta_chart, drawdown_chart, price_chart, var_chart
from data import get_chart_data, get_price_history, search_tickers
from metrics import (
    RETURN_PERIODS,
    TRADING_DAYS_PER_YEAR,
    aligned_returns,
    annualised_return,
    annualised_volatility,
    beta,
    correlation,
    daily_returns,
    drawdown_series,
    excess_kurtosis,
    historical_es,
    historical_var,
    max_drawdown,
    parametric_es,
    parametric_var,
    period_returns,
    sharpe_ratio,
    trailing_window,
)
from summary import build_summary
from views.common import BENCHMARKS, CONFIDENCE_LEVELS, format_return


def show_stock_explorer(window_label: str, risk_free: float | pd.Series) -> None:
    """Tab 1: search for a stock, chart its price, and analyse its risk."""
    query = st.text_input(
        "Search by company name or ticker (for example: Apple, Tesco or HSBA.L)",
        key="search_query",
    )

    if not query:
        # "not query" is True when the box is empty, so there's nothing to show yet.
        st.caption("Type a company name or ticker above to get started.")
        return

    # Remove accidental spaces at either end.
    query = query.strip()

    # --- 1. Search for matching companies ---
    try:
        with st.spinner(f"Searching for '{query}'..."):
            matches = search_tickers(query)
    except Exception:
        # If the search service fails, carry on and treat the text as a ticker.
        st.warning("Company search isn't available right now, so I'll treat your text as a ticker.")
        matches = {}

    # --- 2. Decide which ticker to use ---
    if matches:
        # A drop-down list of matches; look up the ticker for the chosen label.
        chosen = st.selectbox("Choose the one you mean:", list(matches.keys()))
        ticker = matches[chosen]
    else:
        # No matches: assume the user typed an exact ticker.
        ticker = query.upper()

    # --- 3. Download five years of daily prices ---
    try:
        with st.spinner(f"Downloading price history for {ticker}..."):
            prices = get_price_history(ticker)
    except Exception:
        st.error("Couldn't connect to Yahoo Finance. Check your internet connection and try again.")
        return  # leave this function early; the other tabs still work

    if prices.empty:
        st.error(f"No price data found for '{ticker}'. Check the spelling or try the ticker symbol.")
        return

    # --- 4. The latest price ---
    # The last value in the "Close" column is the most recent price.
    latest_close = prices["Close"].iloc[-1]
    st.metric(
        f"{ticker}: latest price ({prices.index[-1]:%d %b %Y})",
        f"{latest_close:,.2f}",
        help=(
            "The most recent daily price from Yahoo Finance, in the stock's own trading "
            "currency. During trading hours this is the latest (possibly delayed) price "
            "rather than the final close."
        ),
    )

    note = f"{len(prices):,} trading days of price history loaded."
    if ticker.endswith(".L"):
        note += " Most London-listed shares are priced in pence (100p = £1)."
    st.caption(note)

    # --- 5. Chart with time-range buttons ---
    # st.segmented_control draws a row of buttons; only one can be selected.
    time_range = st.segmented_control(
        "Time range",
        options=["1D", "1W", "1M", "1Y", "5Y"],
        default="1Y",
        key="chart_range",
    )
    # Clicking the selected button again un-selects it, which gives None
    # (Python's word for "nothing"). If that happens, fall back to 1 year.
    if time_range is None:
        time_range = "1Y"

    try:
        with st.spinner(f"Loading {time_range} prices..."):
            chart_data = get_chart_data(ticker, time_range)
    except Exception:
        chart_data = None

    if chart_data is None or chart_data.empty:
        st.info(
            f"No {time_range} price data is available for {ticker} right now. "
            "Intraday data can be missing before the market opens, so try another range."
        )
    else:
        st.plotly_chart(price_chart(chart_data, ticker, time_range))

        if time_range in ("1D", "1W"):
            st.caption(
                "Intraday prices from Yahoo Finance may be delayed by around 15 minutes. "
                "Times are shown in the exchange's local time."
            )

    # st.expander makes a collapsible section: click its title to open or close it.
    with st.expander("Show the most recent prices"):
        # .copy() makes a separate copy, so tidying it can't affect the original data.
        recent = prices.tail(10).copy()
        # strftime turns each date into neat text, e.g. "25 Sep 2026".
        recent.index = recent.index.strftime("%d %b %Y")
        st.dataframe(recent)

    # --- 6. Action buttons, side by side ---
    left, right = st.columns(2)

    with left:
        # type="primary" makes this the highlighted (coloured) button.
        if st.button(f"Analyse {ticker}", type="primary"):
            # A button is only True for ONE rerun, so we remember
            # which ticker was analysed in session state.
            st.session_state.analysed_ticker = ticker

    with right:
        if st.button(f"Add {ticker} to comparison list"):
            if ticker in st.session_state.watchlist:
                st.info(f"{ticker} is already in your comparison list.")
            else:
                # .append() adds one item to the end of a list.
                st.session_state.watchlist.append(ticker)
                st.success(f"Added {ticker}. See the Compare stocks and Portfolio tabs.")

    # --- 7. Risk analysis (appears after "Analyse" is pressed) ---
    # .get() returns None if "analysed_ticker" hasn't been set yet.
    # The analysis stays visible while you use other controls,
    # and disappears if you switch to a different stock.
    if st.session_state.get("analysed_ticker") == ticker:
        st.divider()
        show_risk_analysis(ticker, prices["Close"], window_label, risk_free)


def show_risk_analysis(
    ticker: str, close: pd.Series, window_label: str, risk_free: float | pd.Series
) -> None:
    """Show a plain-English summary, returns and risk measures for one stock.

    window_label ("1Y" or "5Y") and risk_free come from the sidebar settings.
    """
    st.subheader(f"Risk analysis: {ticker}")
    st.caption(
        "Based on daily total returns (including dividends) in the stock's own currency. "
        "Past performance does not predict future results."
    )

    # An empty bordered box near the top of the section. We fill it in at the
    # END of this function, once every number has been calculated, but it
    # still appears up here on the page.
    summary_box = st.container(border=True)

    # --- Returns over each period ---
    st.markdown("**Returns over each period**")
    returns_by_period = period_returns(close)

    # Make one column per period, so the numbers sit side by side.
    columns = st.columns(len(returns_by_period))

    # zip() pairs things up in order: 1st column with 1st period, 2nd with 2nd, ...
    for column, (label, value) in zip(columns, returns_by_period.items()):
        column.metric(
            label,
            format_return(value),
            help="Total return over this period, including dividends. Not annualised.",
        )

    # Keep only the chosen window of prices, and turn them into daily returns.
    window_prices = trailing_window(close, RETURN_PERIODS[window_label])
    returns = daily_returns(window_prices)

    # The risk-free rate that actually applied over this window, on average.
    risk_free_rate = average_risk_free_rate(risk_free, window_prices.index[0], window_prices.index[-1])

    # --- Calculate the risk measures (the maths lives in metrics.py) ---
    daily_vol = returns.std()
    annual_vol = annualised_volatility(returns)
    annual_ret = annualised_return(returns)
    sharpe = sharpe_ratio(returns, risk_free_rate)
    # max_drawdown returns three values at once; we "unpack" them into three variables.
    worst_fall, peak_date, trough_date = max_drawdown(window_prices)

    st.caption(
        f"Risk measures below use {len(returns):,} daily returns ({window_label} window). "
        f"Sharpe ratios use {risk_free_rate:.2%}, the average risk-free rate over the same "
        "period. You can change both settings in the sidebar."
    )

    # --- Show the three headline measures side by side ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Annualised volatility",
            f"{annual_vol:.1%}",
            help=(
                "How much the return typically varies over a year: the standard "
                "deviation of daily returns, scaled up by the square root of 252 "
                "trading days. Higher means a bumpier ride. It treats rises and "
                "falls the same."
            ),
        )
        st.caption(
            f"Daily {daily_vol:.2%} × √{TRADING_DAYS_PER_YEAR} "
            f"(≈ {np.sqrt(TRADING_DAYS_PER_YEAR):.2f})"
        )

    with col2:
        st.metric(
            "Maximum drawdown",
            f"{worst_fall:.1%}",
            help=(
                "The largest fall from a peak to a later low during the analysis "
                "window: the worst loss for someone who bought at the top and sold "
                "at the bottom."
            ),
        )
        # {date:%d %b %Y} formats a date neatly inside an f-string.
        st.caption(f"Peak {peak_date:%d %b %Y} → low {trough_date:%d %b %Y}")

    with col3:
        st.metric(
            "Sharpe ratio",
            f"{sharpe:.2f}",
            help=(
                "Return earned above the risk-free rate for each unit of volatility: "
                "(annualised return − risk-free rate) ÷ annualised volatility. "
                "Higher means more return per unit of risk. Based on past data only."
            ),
        )
        st.caption(f"({annual_ret:.1%} − {risk_free_rate:.1%}) ÷ {annual_vol:.1%}")

    # --- The "underwater" drawdown chart ---
    st.plotly_chart(drawdown_chart(drawdown_series(window_prices), ticker))

    # --- Value at Risk and Expected Shortfall (1-day horizon) ---
    st.markdown("**Value at Risk (VaR) and Expected Shortfall (ES), 1-day horizon**")

    confidence_label = st.segmented_control(
        "Confidence level",
        options=["95%", "99%"],
        default="95%",
        key="confidence_level",
    )
    if confidence_label is None:
        confidence_label = "95%"
    confidence = CONFIDENCE_LEVELS[confidence_label]
    tail_share = 1 - confidence  # e.g. 0.05: the worst 5% of days

    hist_var = historical_var(returns, confidence)
    hist_es = historical_es(returns, confidence)
    param_var = parametric_var(returns, confidence)
    param_es = parametric_es(returns, confidence)

    var_left, var_right = st.columns(2)

    with var_left:
        st.markdown("*Historical method: uses the actual past returns*")
        st.metric(
            f"{confidence_label} VaR (historical)",
            f"{hist_var:.2%}",
            help=(
                f"The daily loss that was not exceeded on {confidence_label} of past days. "
                "Found by sorting the past daily returns and reading off the cut-off "
                f"for the worst {tail_share:.0%}."
            ),
        )
        st.metric(
            f"{confidence_label} Expected Shortfall (historical)",
            f"{hist_es:.2%}",
            help=(
                f"The average loss on the worst {tail_share:.0%} of past days. "
                "It answers: when a bad day happens, how bad is it on average?"
            ),
        )

    with var_right:
        st.markdown("*Parametric method: assumes a normal distribution*")
        st.metric(
            f"{confidence_label} VaR (parametric)",
            f"{param_var:.2%}",
            help=(
                "VaR from the average and volatility, assuming daily returns follow a "
                "normal (bell-curve) distribution: z × daily volatility − average "
                "daily return, where z is 1.645 at 95% and 2.326 at 99%."
            ),
        )
        st.metric(
            f"{confidence_label} Expected Shortfall (parametric)",
            f"{param_es:.2%}",
            help="The average loss beyond the VaR, assuming a normal distribution.",
        )

    st.caption(
        f"In plain terms: on {confidence_label} of past days, the loss was no more than "
        f"{hist_var:.2%} (about {10000 * hist_var:,.0f} on 10,000 invested). "
        f"On the worst {tail_share:.0%} of days, the average loss was {hist_es:.2%}."
    )

    # Evidence about "fat tails".
    worst_day = returns.min()
    worst_date = returns.idxmin()
    kurtosis = excess_kurtosis(returns)
    # (returns < -param_var) gives True/False for each day; .mean() of True/False
    # values is the share of days that were True.
    breach_rate = (returns < -param_var).mean()

    st.caption(
        f"Worst single day: {worst_day:.2%} on {worst_date:%d %b %Y}. "
        f"Excess kurtosis: {kurtosis:.1f} (0 for a normal distribution; higher means fatter tails). "
        f"Days worse than the parametric VaR: {breach_rate:.1%} "
        f"(a perfect normal model would give {tail_share:.0%})."
    )

    st.plotly_chart(var_chart(returns, hist_var, param_var, ticker, confidence_label))

    # --- Market sensitivity (beta) ---
    st.markdown("**Market sensitivity (beta)**")

    # Suggest the stock's home market: FTSE 100 for London-listed stocks
    # (tickers ending in ".L"), otherwise the S&P 500.
    benchmark_names = list(BENCHMARKS.keys())
    if ticker.endswith(".L"):
        default_position = benchmark_names.index("FTSE 100 (UK)")
    else:
        default_position = benchmark_names.index("S&P 500 (US)")

    # The label includes the ticker, so each stock gets its own sensible default.
    benchmark_name = st.selectbox(
        f"Market index to measure {ticker} against",
        benchmark_names,
        index=default_position,
    )
    benchmark_ticker = BENCHMARKS[benchmark_name]

    try:
        with st.spinner(f"Downloading {benchmark_name} prices..."):
            market_history = get_price_history(benchmark_ticker)
    except Exception:
        market_history = None

    if market_history is None or market_history.empty:
        st.warning(f"Couldn't download {benchmark_name} data, so beta can't be shown right now.")
        # Mark the market-based numbers as missing, so the summary can skip them.
        stock_beta = None
        stock_corr = None
        market_vol = None
    else:
        # Use the same analysis window for the market as for the stock.
        market_prices = trailing_window(market_history["Close"], RETURN_PERIODS[window_label])
        paired = aligned_returns(window_prices, market_prices)

        stock_beta = beta(paired)
        stock_corr = correlation(paired)
        stock_vol = annualised_volatility(paired["stock"])
        market_vol = annualised_volatility(paired["market"])

        beta_left, beta_right = st.columns(2)

        with beta_left:
            st.metric(
                "Beta",
                f"{stock_beta:.2f}",
                help=(
                    "How strongly the stock tends to move with the market. "
                    "1 = in line with the market; above 1 = amplifies market moves; "
                    "below 1 = dampens them. Calculated as covariance ÷ market variance."
                ),
            )
            st.caption(
                f"Correlation {stock_corr:.2f} × (stock volatility {stock_vol:.1%} "
                f"÷ market volatility {market_vol:.1%}) = {stock_beta:.2f}"
            )

        with beta_right:
            st.metric(
                "Correlation with the market",
                f"{stock_corr:.2f}",
                help=(
                    "How closely the stock's daily returns move together with the "
                    "market's, from -1 (opposite) to +1 (perfectly together)."
                ),
            )
            # ** means "to the power of", so stock_corr ** 2 is correlation squared.
            st.caption(
                f"R² = {stock_corr ** 2:.0%}: the share of the stock's daily ups and downs "
                "explained by the market. The rest is specific to the company."
            )

        st.plotly_chart(beta_chart(paired, stock_beta, ticker, benchmark_name))

    # --- Fill in the plain-English summary box at the top ---
    # A dictionary bundles all the numbers together under readable labels.
    stats = {
        "window_label": window_label,
        "return_1y": returns_by_period["1Y"],
        "return_5y": returns_by_period["5Y"],
        "annual_vol": annual_vol,
        "market_vol": market_vol,
        "max_drawdown": worst_fall,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "current_drawdown": drawdown_series(window_prices).iloc[-1],  # today's value
        "sharpe": sharpe,
        "risk_free_rate": risk_free_rate,
        "beta": stock_beta,
        # "X if condition else Y" picks X when the condition is true, otherwise Y.
        "r_squared": stock_corr ** 2 if stock_corr is not None else None,
        "benchmark_name": benchmark_name,
        "confidence": confidence,
        "hist_var": hist_var,
        "hist_es": hist_es,
        "param_es": param_es,
        "kurtosis": kurtosis,
    }

    # "with summary_box:" writes into the box created near the top of this function.
    with summary_box:
        st.markdown("**In plain English**")
        for sentence in build_summary(ticker, stats):
            st.markdown(f"- {sentence}")
