# app.py
# The main file of the Stock Risk & Portfolio Analyser.
# Streamlit runs this file from top to bottom to build the web page.
#
# How this file is organised:
#   1. Imports and page settings
#   2. Small helper functions (formatting, caching)
#   3. One function per part of the page:
#        show_risk_analysis   - the detailed risk analysis of one stock
#        show_stock_explorer  - tab 1: search, price chart, "Analyse" button
#        show_comparison      - tab 2: compare stocks rescaled to 100
#        show_portfolio       - tab 3: portfolio suggestion and efficient frontier
#        show_monte_carlo     - tab 3 (continued): Monte Carlo projection
#   4. The page itself: sidebar settings, title, disclaimer and tabs

import numpy as np
import pandas as pd
import streamlit as st

# Import our own functions from the other files in this folder.
# Brackets let a long import list continue over several lines.
from charts import (
    beta_chart,
    comparison_chart,
    correlation_heatmap,
    drawdown_chart,
    efficient_frontier_chart,
    fan_chart,
    price_chart,
    var_chart,
    weights_chart,
)
from data import get_aligned_closes, get_chart_data, get_price_history, search_tickers
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
from portfolio import (
    RISK_AVERSION,
    correlation_matrix,
    daily_returns_table,
    efficient_frontier,
    estimate_inputs,
    individual_volatilities,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_return,
    portfolio_sharpe,
    portfolio_volatility,
    random_portfolios,
    risk_appetite_weights,
    summary_table,
)
from montecarlo import (
    HORIZON_YEARS,
    bootstrap_paths,
    exact_percentile,
    final_value_summary,
    monthly_portfolio_returns,
    percentile_bands,
    simulate_paths,
)
from summary import build_summary

# st.set_page_config sets the browser-tab title and icon. It must be the first
# Streamlit command on the page.
st.set_page_config(page_title="Stock Risk & Portfolio Analyser", page_icon="📈")

# Market indices available for beta. On Yahoo Finance, "^" marks an index.
BENCHMARKS = {
    "S&P 500 (US)": "^GSPC",
    "FTSE 100 (UK)": "^FTSE",
    "Nasdaq 100 (US)": "^NDX",
    "Euro Stoxx 50 (Europe)": "^STOXX50E",
}

# Confidence levels for VaR and Expected Shortfall.
CONFIDENCE_LEVELS = {"95%": 0.95, "99%": 0.99}

# The disclaimer shown at the top of every page.
DISCLAIMER = (
    "**Educational tool, not financial advice.** This app describes how shares behaved "
    "in the past and illustrates portfolio theory. It never recommends buying or selling "
    "anything, and past performance does not predict future results."
)


# ===========================================================================
# 2. HELPER FUNCTIONS
# ===========================================================================
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


@st.cache_data(show_spinner=False)
def cached_frontier_and_cloud(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    max_weight: float,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The efficient frontier and the cloud of random portfolios, remembered between reruns.

    Tracing the frontier runs the optimiser 40 times, so we cache it: clicking a button
    elsewhere on the page won't recalculate it unless these inputs change.
    """
    frontier = efficient_frontier(expected_returns, covariance, max_weight)
    cloud = random_portfolios(expected_returns, covariance, risk_free_rate)
    return frontier, cloud


# ===========================================================================
# 3. ONE FUNCTION PER PART OF THE PAGE
# ===========================================================================
def show_risk_analysis(
    ticker: str, close: pd.Series, window_label: str, risk_free_rate: float
) -> None:
    """Show a plain-English summary, returns and risk measures for one stock.

    window_label ("1Y" or "5Y") and risk_free_rate come from the sidebar settings.
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

    # --- Calculate the risk measures (the maths lives in metrics.py) ---
    daily_vol = returns.std()
    annual_vol = annualised_volatility(returns)
    annual_ret = annualised_return(returns)
    sharpe = sharpe_ratio(returns, risk_free_rate)
    # max_drawdown returns three values at once; we "unpack" them into three variables.
    worst_fall, peak_date, trough_date = max_drawdown(window_prices)

    st.caption(
        f"Risk measures below use {len(returns):,} daily returns ({window_label} window) "
        f"and a {risk_free_rate:.2%} risk-free rate. You can change both in the sidebar."
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


def show_stock_explorer(window_label: str, risk_free_rate: float) -> None:
    """Tab 1: search for a stock, chart its price, and analyse its risk."""
    query = st.text_input(
        "Search by company name or ticker (for example: Apple, Tesco or HSBA.L)"
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
        f"{ticker}: latest closing price",
        f"{latest_close:,.2f}",
        help="In the stock's own trading currency.",
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
        show_risk_analysis(ticker, prices["Close"], window_label, risk_free_rate)


def show_comparison() -> None:
    """Tab 2: compare the stocks in the comparison list, each rescaled to start at 100."""
    st.subheader("Compare stocks")
    st.write(
        "Each stock is rescaled to start at 100, so you can compare performance "
        "regardless of share price or currency."
    )

    watchlist = st.session_state.watchlist

    # len() counts the items in a list.
    if len(watchlist) == 0:
        st.info(
            "Your comparison list is empty. In the Explore a stock tab, search for a "
            "stock and click 'Add ... to comparison list'."
        )
        return

    # ", ".join(...) glues the list items together into one piece of text.
    st.write("**In your list:** " + ", ".join(watchlist))

    if st.button("Clear comparison list"):
        st.session_state.watchlist = []
        st.rerun()  # start a fresh run straight away so the page updates

    if len(watchlist) == 1:
        st.caption("Add at least one more stock to compare.")

    compare_range = st.segmented_control(
        "Comparison period",
        options=["1M", "1Y", "5Y"],
        default="1Y",
        key="compare_range",
    )
    if compare_range is None:
        compare_range = "1Y"

    try:
        with st.spinner("Building the comparison..."):
            closes = get_aligned_closes(watchlist, compare_range)
    except Exception:
        st.error("Couldn't download the comparison data. Check your internet connection and try again.")
        return

    if closes.empty:
        st.warning("Couldn't find overlapping price data for these stocks.")
        return

    st.plotly_chart(comparison_chart(closes, compare_range))


def show_portfolio(window_label: str, risk_free_rate: float) -> None:
    """Tab 3: build a portfolio from the comparison list, optimise it, and project it."""
    st.subheader("Portfolio suggestion")
    st.write(
        "Builds a portfolio from the stocks in your comparison list using "
        "mean-variance analysis (Modern Portfolio Theory). Estimates come from "
        "past data, in each stock's own currency; they are not forecasts."
    )

    watchlist = st.session_state.watchlist
    if len(watchlist) < 2:
        st.info(
            "Add at least two stocks to your comparison list (in the Explore a stock tab) "
            "to build a portfolio."
        )
        return

    try:
        with st.spinner("Estimating returns and risks..."):
            portfolio_closes = get_aligned_closes(watchlist, window_label)
    except Exception:
        st.error("Couldn't download the data for the portfolio. Check your internet connection and try again.")
        return

    # We need at least 2 stocks with data, and at least ~3 months of shared history.
    if len(portfolio_closes.columns) < 2 or len(portfolio_closes) < 60:
        st.warning("There isn't enough overlapping price history for these stocks to build a portfolio.")
        return

    st.caption(
        f"Estimates use the last {window_label} of daily data and a {risk_free_rate:.2%} "
        "risk-free rate. You can change both in the sidebar."
    )

    # The two inputs to mean-variance analysis (the maths lives in portfolio.py).
    expected_returns, covariance = estimate_inputs(portfolio_closes)
    volatilities = individual_volatilities(covariance)
    tickers = expected_returns.index  # the stock names, in order

    # --- 1. Each stock on its own ---
    st.markdown("**Each stock on its own (estimated from past data)**")
    stock_table = pd.DataFrame(
        {
            "Average annual return": expected_returns,
            "Annual volatility": volatilities,
        }
    )
    # .style.format("{:.1%}") shows every number as a percentage with 1 decimal place.
    st.dataframe(stock_table.style.format("{:.1%}"))

    # --- 2. How the stocks move together ---
    st.plotly_chart(correlation_heatmap(correlation_matrix(portfolio_closes)))

    # --- 3. A simple equal-weight portfolio ---
    n_stocks = len(expected_returns)
    # np.full(3, 1/3) makes an array [1/3, 1/3, 1/3]: equal weights that add up to 1.
    equal_weights = np.full(n_stocks, 1 / n_stocks)

    ew_return = portfolio_return(equal_weights, expected_returns)
    ew_vol = portfolio_volatility(equal_weights, covariance)
    ew_sharpe = portfolio_sharpe(equal_weights, expected_returns, covariance, risk_free_rate)

    st.markdown(f"**Equal-weight portfolio ({1 / n_stocks:.0%} in each stock)**")
    e1, e2, e3 = st.columns(3)
    e1.metric(
        "Expected return (annual)",
        f"{ew_return:.1%}",
        help="The weighted average of each stock's average annual return.",
    )
    e2.metric(
        "Volatility (annual)",
        f"{ew_vol:.1%}",
        help="Portfolio volatility, which depends on how the stocks move together.",
    )
    e3.metric(
        "Sharpe ratio",
        f"{ew_sharpe:.2f}",
        help="(Portfolio return − risk-free rate) ÷ portfolio volatility.",
    )

    # Diversification: compare the portfolio's volatility with the
    # weighted average of the individual volatilities.
    average_vol = float(equal_weights @ volatilities.values)
    reduction = 1 - ew_vol / average_vol
    st.caption(
        f"Diversification at work: the individual stocks' volatilities average "
        f"{average_vol:.1%}, but the portfolio's is only {ew_vol:.1%}, about "
        f"{reduction:.0%} lower, because the stocks don't move perfectly together."
    )

    # --- 4. Optimised portfolio for the chosen risk appetite ---
    st.markdown("**Optimised portfolio for your risk appetite**")

    o_left, o_right = st.columns(2)

    with o_left:
        appetite = st.segmented_control(
            "Risk appetite",
            options=["Low", "Medium", "High"],
            default="Medium",
            key="risk_appetite",
            help=(
                "Low favours steadier stocks; high accepts more volatility in "
                "exchange for a higher expected return. Technically, the optimiser "
                "maximises: expected return − ½ × A × variance, with risk aversion "
                "A = 10 (low), 4 (medium) or 1.5 (high)."
            ),
        )

    with o_right:
        max_weight_pct = st.number_input(
            "Maximum in any one stock (%)",
            min_value=10,
            max_value=100,
            value=100,
            step=5,
            key="max_weight_pct",
            help=(
                "A cap on any single stock's weight. 100% means no cap. "
                "Caps stop the optimiser putting everything into one stock."
            ),
        )

    if appetite is None:
        appetite = "Medium"

    # The weights must add up to 100%, so with 3 stocks a cap below 34% is
    # impossible. max() picks the larger of the two numbers.
    max_weight = max(max_weight_pct / 100, 1 / n_stocks)
    if max_weight_pct / 100 < 1 / n_stocks:
        st.caption(
            f"With {n_stocks} stocks, the cap can't be below {1 / n_stocks:.0%} "
            f"(the weights must add up to 100%), so {1 / n_stocks:.0%} is used."
        )

    try:
        with st.spinner("Optimising..."):
            suggested = risk_appetite_weights(
                expected_returns, covariance, RISK_AVERSION[appetite], max_weight
            )
            min_var = min_variance_weights(covariance, max_weight)
            best_sharpe = max_sharpe_weights(expected_returns, covariance, risk_free_rate, max_weight)
    except Exception:
        st.error("The optimiser couldn't find a solution for these settings. Try a different cap.")
        return  # leave this function early; nothing more to show

    s_return = portfolio_return(suggested, expected_returns)
    s_vol = portfolio_volatility(suggested, covariance)
    s_sharpe = portfolio_sharpe(suggested, expected_returns, covariance, risk_free_rate)

    # Bar chart of the suggested split. pd.Series(...) labels each weight with its ticker.
    st.plotly_chart(
        weights_chart(
            pd.Series(suggested, index=tickers),
            f"Suggested split for a {appetite.lower()} risk appetite",
        )
    )

    # delta= adds a small coloured change underneath each number.
    # delta_or_none hides it when the change rounds to zero.
    s1, s2, s3 = st.columns(3)
    s1.metric(
        "Expected return (annual)",
        f"{s_return:.1%}",
        delta=delta_or_none(
            s_return - ew_return, f"{s_return - ew_return:+.1%} vs equal weight", 3
        ),
    )
    s2.metric(
        "Volatility (annual)",
        f"{s_vol:.1%}",
        delta=delta_or_none(s_vol - ew_vol, f"{s_vol - ew_vol:+.1%} vs equal weight", 3),
        # "inverse": a FALL in volatility shows green, a rise shows red.
        delta_color="inverse",
    )
    s3.metric(
        "Sharpe ratio",
        f"{s_sharpe:.2f}",
        delta=delta_or_none(
            s_sharpe - ew_sharpe, f"{s_sharpe - ew_sharpe:+.2f} vs equal weight", 2
        ),
    )

    # If the answer is (almost) the same as equal weights, say so plainly.
    # np.abs(...).max() is the biggest difference between any two matching weights.
    if np.abs(suggested - equal_weights).max() < 0.005:
        st.caption(
            "For these stocks and this risk appetite, the optimiser's answer is "
            "(almost exactly) an equal split."
        )

    st.caption(
        "This is a mathematical illustration based on past data, for education only. "
        "It is not a recommendation to buy or sell anything, and past average returns "
        "are a poor guide to future returns."
    )

    # --- 5. Compare several portfolios side by side ---
    with st.expander("Compare with other portfolios"):
        portfolios = {
            "Equal weight": equal_weights,
            "Minimum variance": min_var,
            "Maximum Sharpe ratio": best_sharpe,
            f"Suggested ({appetite} risk appetite)": suggested,
        }
        comparison = summary_table(portfolios, expected_returns, covariance, risk_free_rate)

        # Show everything as percentages, except the Sharpe ratio (2 decimals).
        st.dataframe(
            comparison.style.format("{:.1%}").format("{:.2f}", subset=["Sharpe ratio"])
        )
        st.caption(
            "Minimum variance: the lowest-risk mix (it ignores expected returns). "
            "Maximum Sharpe ratio: the best return per unit of risk."
        )

    # --- 6. The efficient frontier ---
    st.markdown("**The efficient frontier**")

    try:
        with st.spinner("Tracing the efficient frontier..."):
            # Cached (see cached_frontier_and_cloud above), so it only recalculates
            # when the stocks or settings change.
            frontier, cloud = cached_frontier_and_cloud(
                expected_returns, covariance, max_weight, risk_free_rate
            )
    except Exception:
        frontier = None

    if frontier is None or frontier.empty:
        st.warning("Couldn't trace the efficient frontier for these settings.")
    else:
        # Each individual stock as a point: (volatility, expected return).
        stock_points = pd.DataFrame(
            {"Volatility": volatilities, "Expected return": expected_returns}
        )

        # The portfolios to mark with stars: name -> (volatility, return).
        highlights = {
            "Equal weight": (ew_vol, ew_return),
            "Minimum variance": (
                portfolio_volatility(min_var, covariance),
                portfolio_return(min_var, expected_returns),
            ),
            "Maximum Sharpe ratio": (
                portfolio_volatility(best_sharpe, covariance),
                portfolio_return(best_sharpe, expected_returns),
            ),
            f"Suggested ({appetite})": (s_vol, s_return),
        }

        st.plotly_chart(
            efficient_frontier_chart(frontier, cloud, stock_points, highlights, risk_free_rate)
        )
        st.caption(
            "How to read it: each small dot is one possible mix of these stocks. The red "
            "curve is the efficient frontier: for each level of risk, the highest expected "
            "return available. Anything below the curve is 'inefficient', because another "
            "mix offers more return for the same risk. Diamonds are single stocks; stars "
            "are the portfolios above. The dashed line mixes cash with the maximum-Sharpe "
            "portfolio and touches the curve exactly at that portfolio."
        )

    # --- 7. Monte Carlo projection of the suggested portfolio ---
    show_monte_carlo(portfolio_closes, suggested, s_return, s_vol, appetite)


def show_monte_carlo(
    portfolio_closes: pd.DataFrame,
    suggested: np.ndarray,
    s_return: float,
    s_vol: float,
    appetite: str,
) -> None:
    """Tab 3 (continued): simulate 10,000 futures for the suggested portfolio."""
    st.divider()
    st.markdown("**Monte Carlo projection of the suggested portfolio**")
    st.write(
        "Simulates 10,000 possible futures for the suggested portfolio. "
        "The shaded band shows the range between the 5th and 95th percentiles: "
        "in 90% of the simulations, the value ended up inside it."
    )

    mc_left, mc_middle, mc_right = st.columns(3)

    with mc_left:
        horizon = st.segmented_control(
            "Projection horizon",
            options=list(HORIZON_YEARS.keys()),  # ["1Y", "5Y", "10Y"]
            default="5Y",
            key="mc_horizon",
        )

    with mc_middle:
        method = st.segmented_control(
            "Simulation method",
            options=["Normal model", "Historical bootstrap"],
            default="Normal model",
            key="mc_method",
            help=(
                "Normal model: monthly log-returns drawn from a normal distribution "
                "with the portfolio's average return and volatility. "
                "Historical bootstrap: each simulated month re-uses a real past month "
                "of the portfolio's returns, picked at random, so real fat tails carry through."
            ),
        )

    with mc_right:
        start_value = st.number_input(
            "Amount invested",
            min_value=100,
            max_value=10_000_000,
            value=10_000,
            step=1_000,
            key="mc_start_value",
            help=(
                "An illustrative amount. The stocks may trade in different "
                "currencies; currency movements are not modelled."
            ),
        )

    if horizon is None:
        horizon = "5Y"
    if method is None:
        method = "Normal model"
    years = HORIZON_YEARS[horizon]

    # Run BOTH methods (the maths lives in montecarlo.py), so we can compare them.
    with st.spinner("Simulating 10,000 possible futures..."):
        # Method 1: the normal model, using the suggested portfolio's average and volatility.
        normal_paths = simulate_paths(start_value, s_return, s_vol, years)

        # Method 2: the historical bootstrap, using the portfolio's real past monthly returns.
        past_monthly = monthly_portfolio_returns(daily_returns_table(portfolio_closes), suggested)
        bootstrap_result = bootstrap_paths(start_value, past_monthly, years)

    # Show the chart and numbers for whichever method is selected.
    if method == "Normal model":
        paths = normal_paths
    else:
        paths = bootstrap_result

    bands = percentile_bands(paths, years)
    final = final_value_summary(paths, start_value)

    # paths[:20] = the first 20 rows, i.e. 20 example futures to draw faintly.
    st.plotly_chart(
        fan_chart(
            bands,
            paths[:20],
            start_value,
            f"{method}: 10,000 simulated futures over {years} year(s), "
            f"{appetite.lower()} risk appetite portfolio",
        )
    )

    # The key numbers at the end of the horizon, in a 2 x 2 grid so the labels fit.
    row1_left, row1_right = st.columns(2)
    row2_left, row2_right = st.columns(2)
    row1_left.metric(
        "Bad case (5th percentile)",
        f"{final['5th']:,.0f}",
        help="1 in 20 simulated futures ended below this value.",
    )
    row1_right.metric(
        "Median (50th percentile)",
        f"{final['50th']:,.0f}",
        help="Half the simulated futures ended above this value, and half below.",
    )
    row2_left.metric(
        "Good case (95th percentile)",
        f"{final['95th']:,.0f}",
        help="1 in 20 simulated futures ended above this value.",
    )
    row2_right.metric(
        "Chance of ending below the amount invested",
        f"{final['chance_of_loss']:.0%}",
        help="The share of the 10,000 simulated futures that finished below the starting amount.",
    )

    if method == "Normal model":
        # Check the simulation against the exact formula for the median.
        exact_median = exact_percentile(start_value, s_return, s_vol, years, 50)
        st.caption(
            f"Check: under this model the median has an exact formula, which gives "
            f"{exact_median:,.0f}. The simulation gives {final['50th']:,.0f}, so the two agree "
            "closely. With more simulations, they would agree even more closely."
        )
    else:
        st.caption(
            f"The bootstrap re-uses {len(past_monthly)} months of the portfolio's actual "
            "past returns, picked at random with replacement."
        )
        if len(past_monthly) < 36:
            st.warning(
                "Fewer than 3 years of monthly history are being re-used, so the bootstrap "
                "is drawing from very few outcomes. Try the 5Y history setting above."
            )

    # --- Compare the two methods side by side: this is "model risk" ---
    st.markdown("**How much does the choice of model matter?**")
    normal_final = final_value_summary(normal_paths, start_value)
    bootstrap_final = final_value_summary(bootstrap_result, start_value)

    method_table = pd.DataFrame(
        {
            "Bad case (5th)": [normal_final["5th"], bootstrap_final["5th"]],
            "Median (50th)": [normal_final["50th"], bootstrap_final["50th"]],
            "Good case (95th)": [normal_final["95th"], bootstrap_final["95th"]],
            "Chance of a loss": [normal_final["chance_of_loss"], bootstrap_final["chance_of_loss"]],
        },
        index=["Normal model", "Historical bootstrap"],
    )
    # A dictionary of formats: money columns with commas, the chance as a percentage.
    st.dataframe(
        method_table.style.format(
            {
                "Bad case (5th)": "{:,.0f}",
                "Median (50th)": "{:,.0f}",
                "Good case (95th)": "{:,.0f}",
                "Chance of a loss": "{:.0%}",
            }
        )
    )
    st.caption(
        "Both methods use the same past data, but make different assumptions. The gap "
        "between them is a measure of model risk: how much the answer depends on the "
        "modelling choice rather than on the data."
    )

    st.caption(
        "These projections assume the past continues: constant weights (rebalanced), no fees, "
        "no inflation and no currency effects. The average return itself is a very uncertain "
        "estimate. This is an educational illustration, not a forecast or a recommendation."
    )


# ===========================================================================
# 4. THE PAGE ITSELF
# ===========================================================================

# SESSION STATE: memory that survives reruns (for as long as this tab is open).
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# --- The sidebar: settings shared by every tab, plus "About" ---
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

    # st.number_input draws a box for typing a number, with + and - buttons.
    risk_free_pct = st.number_input(
        "Risk-free rate (% per year)",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.25,
        key="risk_free_pct",
        help=(
            "The return available with (almost) no risk, such as short-term government "
            "bills. Used for Sharpe ratios. 4% is a round-number assumption: change it "
            "to match current interest rates."
        ),
    )

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
risk_free_rate = risk_free_pct / 100  # convert e.g. 4.0 (%) into 0.04

# --- Title and disclaimer ---
st.title("Stock Risk & Portfolio Analyser")
st.caption("Search for a stock, measure its risk, compare stocks and explore a portfolio.")
st.info(DISCLAIMER, icon="ℹ️")

# --- The three tabs ---
# st.tabs returns one container per tab; "with" puts content inside it.
tab_explore, tab_compare, tab_portfolio = st.tabs(
    ["Explore a stock", "Compare stocks", "Portfolio & projection"]
)

with tab_explore:
    show_stock_explorer(window_label, risk_free_rate)

with tab_compare:
    show_comparison()

with tab_portfolio:
    show_portfolio(window_label, risk_free_rate)

# --- Footer ---
st.divider()
st.caption(
    "Educational tool, not financial advice. Figures describe the past and use free data "
    "that may be delayed or incomplete. Currency effects, fees and taxes are not modelled."
)