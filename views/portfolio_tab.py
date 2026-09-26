# views/portfolio_tab.py
# Tab 3, "Portfolio & projection" (part 1): build a portfolio from the comparison list,
# optimise it for a risk appetite, and draw the efficient frontier.
# The goal planner at the bottom of the tab lives in views/goal_planner.py.

import numpy as np
import pandas as pd
import streamlit as st

from backtest import average_risk_free_rate
from charts import correlation_heatmap, efficient_frontier_chart, weights_chart
from data import get_aligned_closes
from portfolio import (
    RISK_AVERSION,
    correlation_matrix,
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
from views.common import delta_or_none, latest_risk_free_rate
from views.goal_planner import show_goal_planner


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


def show_portfolio(window_label: str, risk_free: float | pd.Series, inflation: float) -> None:
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

    # Today's rate, for the goal planner's forward-looking return assumption.
    todays_risk_free = latest_risk_free_rate(risk_free)

    # The risk-free rate that actually applied over the estimation window, on average.
    risk_free_rate = average_risk_free_rate(
        risk_free, portfolio_closes.index[0], portfolio_closes.index[-1]
    )
    st.caption(
        f"Estimates use the last {window_label} of daily data. Sharpe ratios use "
        f"{risk_free_rate:.2%}, the average risk-free rate over the same period. You can "
        "change both settings in the sidebar."
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

    # --- 7. The goal planner (a Monte Carlo projection), in views/goal_planner.py ---
    show_goal_planner(portfolio_closes, suggested, s_return, s_vol, appetite, inflation, todays_risk_free)
