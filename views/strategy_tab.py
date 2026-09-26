# views/strategy_tab.py
# Tab 4, "Strategy test": an honest, out-of-sample (walk-forward) test of six
# portfolio strategies, with trading costs.

import pandas as pd
import streamlit as st

from backtest import STRATEGIES, average_risk_free_rate, rebalance_dates, run_all_strategies
from charts import backtest_chart
from data import get_full_aligned_closes
from views.common import LOOKBACK_YEARS, REBALANCE_MONTHS


@st.cache_data(show_spinner=False)
def cached_backtest(
    returns: pd.DataFrame,
    lookback_years: int,
    rebalance_months: int,
    cost_rate: float,
    risk_free: float | pd.Series,
    max_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The walk-forward test of every strategy, remembered between reruns.

    It runs hundreds of optimisations, so it's only recalculated when an input changes.
    """
    return run_all_strategies(
        returns, lookback_years, rebalance_months, cost_rate, risk_free, max_weight
    )


def show_strategy_test(risk_free: float | pd.Series) -> None:
    """Tab 4: an honest, out-of-sample (walk-forward) test of portfolio strategies."""
    st.subheader("Strategy test: how did each approach do on data it hadn't seen?")
    st.write(
        "Every portfolio in the other tabs is chosen using the same history it's then "
        "judged on, so it always looks good in hindsight. This test avoids that. At each "
        "rebalance date, every strategy chooses its weights using **only the past**, then "
        "holds them for the next period, which it never saw. Trading costs are charged "
        "whenever the weights change. It uses up to 10 years of prices."
    )

    watchlist = st.session_state.watchlist
    if len(watchlist) < 2:
        st.info(
            "Add at least two stocks to your comparison list (in the Explore a stock tab) "
            "to run the strategy test."
        )
        return

    # --- Settings, in a 2 x 2 grid ---
    row1_left, row1_right = st.columns(2)
    row2_left, row2_right = st.columns(2)

    with row1_left:
        lookback_label = st.segmented_control(
            "Look-back window (history used to choose weights)",
            options=list(LOOKBACK_YEARS.keys()),
            default="2Y",
            key="bt_lookback",
        )
    with row1_right:
        rebalance_label = st.segmented_control(
            "How often to rebalance",
            options=list(REBALANCE_MONTHS.keys()),
            default="Every year",
            key="bt_rebalance",
        )
    with row2_left:
        cost_pct = st.number_input(
            "Trading cost (% of each amount traded)",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.05,
            key="bt_cost",
            help=(
                "Covers dealing charges and bid-ask spreads. Charged on every buy and sell "
                "at each rebalance, including the first purchase."
            ),
        )
    with row2_right:
        cap_pct = st.number_input(
            "Maximum in any one stock (%)",
            min_value=10,
            max_value=100,
            value=100,
            step=5,
            key="bt_max_weight",
            help="100% means no cap. Caps make the optimised strategies less extreme.",
        )

    if lookback_label is None:
        lookback_label = "2Y"
    if rebalance_label is None:
        rebalance_label = "Every year"
    lookback_years = LOOKBACK_YEARS[lookback_label]
    rebalance_months = REBALANCE_MONTHS[rebalance_label]
    max_weight = max(cap_pct / 100, 1 / len(watchlist))  # a cap can't make 100% impossible

    # --- Data: up to 10 years of prices, lined up by date ---
    try:
        with st.spinner("Downloading up to 10 years of prices..."):
            closes = get_full_aligned_closes(watchlist, "10y")
    except Exception:
        st.error("Couldn't download the long price history. Check your internet connection and try again.")
        return

    if len(closes.columns) < 2:
        st.warning("Couldn't find price data for at least two of these stocks.")
        return

    returns = closes.pct_change().dropna()
    history_years = (returns.index[-1] - returns.index[0]).days / 365.25

    # We need the look-back window plus at least one full year to test on.
    if history_years < lookback_years + 1:
        st.warning(
            f"These stocks only share {history_years:.1f} years of history, which isn't enough "
            f"for a {lookback_years}-year look-back plus a test period. Try a shorter look-back, "
            "or remove a recently listed stock."
        )
        return

    # --- Run every strategy (cached, because it's hundreds of optimisations) ---
    try:
        with st.spinner("Testing six strategies through time..."):
            wealth, summary, histories = cached_backtest(
                returns, lookback_years, rebalance_months, cost_pct / 100, risk_free, max_weight
            )
    except Exception:
        st.error("The strategy test couldn't be completed for these settings. Try different settings.")
        return

    n_rebalances = len(rebalance_dates(returns, lookback_years, rebalance_months))
    test_years = (wealth.index[-1] - wealth.index[0]).days / 365.25
    st.caption(
        f"Test period: {wealth.index[0]:%b %Y} to {wealth.index[-1]:%b %Y} "
        f"({test_years:.1f} years that no strategy saw when choosing its weights), "
        f"with {n_rebalances} rebalances. Sharpe and Sortino ratios use "
        f"{average_risk_free_rate(risk_free, wealth.index[0], wealth.index[-1]):.2%}, the average "
        "risk-free rate over the test period; each strategy's choices used only the rates "
        "known at the time."
    )

    # --- The chart ---
    st.plotly_chart(backtest_chart(wealth))

    # --- The results table, sorted by Sharpe ratio (best first) ---
    st.markdown("**Results over the test period (after trading costs)**")
    results = summary.sort_values("Sharpe ratio", ascending=False)
    st.dataframe(
        results.style.format(
            {
                "Annual growth (compound)": "{:.1%}",
                "Volatility": "{:.1%}",
                "Sharpe ratio": "{:.2f}",
                "Sortino ratio": "{:.2f}",
                "Maximum drawdown": "{:.1%}",
                "Expected Shortfall (95%)": "{:.2%}",
                "Final value (from 100)": "{:,.0f}",
                "Average turnover per rebalance": "{:.0%}",
            }
        )
    )

    # How did each strategy do against the equal-weight benchmark?
    benchmark_sharpe = summary.loc["Equal weight", "Sharpe ratio"]
    beat_benchmark = []
    for name in summary.index:
        if name != "Equal weight" and summary.loc[name, "Sharpe ratio"] > benchmark_sharpe:
            beat_benchmark.append(name)

    if beat_benchmark:
        st.caption(
            "On a risk-adjusted basis (Sharpe ratio), these beat simple equal weighting in this "
            "test: " + "; ".join(beat_benchmark) + "."
        )
    else:
        st.caption(
            "None of the optimised strategies beat simple equal weighting on a risk-adjusted "
            "basis (Sharpe ratio) in this test. That's a common finding: estimation error in "
            "the optimiser's inputs often outweighs the benefit of optimising."
        )

    # --- Transparency: the weights each strategy chose ---
    with st.expander("See the weights each strategy chose at every rebalance"):
        chosen = st.selectbox("Strategy", list(STRATEGIES.keys()), key="bt_history_strategy")
        st.dataframe(histories[chosen].style.format("{:.0%}"))
        st.caption(
            "Turnover is the total of the weight changes at that date (buys plus sells). "
            "The first row includes the initial purchase from cash."
        )

    st.caption(
        "How to read this: each line only uses information that was available at the time, so "
        "this is a fairer test than the in-sample results elsewhere in the app. It is still "
        "the past, though. A few years is a small sample, the results depend on the settings, "
        "and the stocks were chosen today, knowing how they turned out (hindsight in stock "
        "selection). It is an educational test, not a recommendation."
    )
