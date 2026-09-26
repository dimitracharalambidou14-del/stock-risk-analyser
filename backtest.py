# backtest.py
# An honest "walk-forward" test of portfolio strategies.
#
# The problem: if you choose a portfolio using some history and then judge it on
# that SAME history, it will always look good in hindsight. This is called
# "look-ahead" or "in-sample" bias.
#
# A walk-forward test avoids it. At each rebalancing date, every strategy chooses
# its weights using ONLY the past (the look-back window), then holds them until
# the next rebalancing date: a period it never saw. Trading costs are charged
# whenever the weights change.
#
# Like metrics.py and portfolio.py, there is no Streamlit code here.

import numpy as np
import pandas as pd

from metrics import TRADING_DAYS_PER_YEAR
from portfolio import max_sharpe_weights, min_variance_weights, optimise

# Expected Shortfall looks at the worst 5% of days.
TAIL_PROBABILITY = 0.05


def average_risk_free_rate(risk_free: float | pd.Series, start, end) -> float:
    """The risk-free rate that actually applied between two dates, on average.

    risk_free: either one fixed rate (e.g. 0.04), or a Series of daily rates
    (dates as the index), e.g. the Bank of England's Bank Rate history.
    If it's a Series, we average the rates between start and end, so a Sharpe
    ratio for 2021-2026 uses the rates of 2021-2026, not just today's rate.
    """
    # isinstance checks what type of thing a variable is.
    if not isinstance(risk_free, pd.Series):
        return float(risk_free)

    period = risk_free[(risk_free.index >= start) & (risk_free.index <= end)]
    if period.empty:
        # No history for those dates: use the most recent rate we have.
        return float(risk_free.iloc[-1])
    return float(period.mean())


# ---------------------------------------------------------------------------
# 1. Extra risk measures, calculated from an array of daily returns
# ---------------------------------------------------------------------------


def sortino_ratio(daily: np.ndarray, risk_free_rate: float) -> float:
    """Like the Sharpe ratio, but only days BELOW the risk-free rate count as risk.

    Sortino = (annual average return - risk-free rate) / annual downside deviation
    Downside deviation is the square root of the average squared SHORTFALL,
    where days above the risk-free rate count as a shortfall of zero.
    """
    daily_risk_free = risk_free_rate / TRADING_DAYS_PER_YEAR
    # np.minimum(x, 0) keeps negative numbers and turns positive ones into 0.
    shortfall = np.minimum(daily - daily_risk_free, 0)
    downside = np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside == 0:
        return 0.0  # no bad days at all: avoid dividing by zero
    excess = np.mean(daily) * TRADING_DAYS_PER_YEAR - risk_free_rate
    return excess / downside


def expected_shortfall(daily: np.ndarray, tail: float = TAIL_PROBABILITY) -> float:
    """The average loss on the worst 5% of days, as a positive number (0.03 = 3%)."""
    cutoff = np.quantile(daily, tail)
    return -np.mean(daily[daily <= cutoff])


def max_drawdown(daily: np.ndarray) -> float:
    """The largest peak-to-trough fall, as a negative number (-0.25 = -25%)."""
    # Start the wealth path at 1, so a fall on the very first day also counts.
    wealth = np.concatenate([[1.0], np.cumprod(1 + daily)])
    # np.maximum.accumulate = the highest value so far (like pandas' cummax).
    running_peak = np.maximum.accumulate(wealth)
    return np.min(wealth / running_peak - 1)


# ---------------------------------------------------------------------------
# 2. A multi-start optimiser for "bumpy" objectives
# ---------------------------------------------------------------------------


def optimise_multistart(objective, n_stocks: int, max_weight: float, n_random_starts: int = 4) -> np.ndarray:
    """Run the optimiser from several starting points and keep the best answer.

    Objectives like maximum drawdown and Expected Shortfall depend on a handful of
    extreme days, so they're "bumpy" and an optimiser can get stuck on a poor
    answer. Trying several starting points makes that much less likely.
    If every attempt fails, fall back to equal weights.
    """
    rng = np.random.default_rng(0)  # fixed seed, so results are reproducible
    starting_points = [np.full(n_stocks, 1 / n_stocks)]
    for _ in range(n_random_starts):
        random_weights = rng.dirichlet(np.ones(n_stocks))  # positive, adding up to 1
        starting_points.append(random_weights)

    best_weights = None
    best_value = np.inf  # np.inf = infinity, so any real answer is better

    for start in starting_points:
        try:
            weights = optimise(objective, n_stocks, max_weight, starting_guess=start)
        except RuntimeError:
            continue  # this attempt failed; try the next starting point
        value = objective(weights)
        if value < best_value:
            best_weights, best_value = weights, value

    if best_weights is None:
        return np.full(n_stocks, 1 / n_stocks)
    return best_weights


# ---------------------------------------------------------------------------
# 3. The strategies. Each takes the PAST daily returns (one column per stock)
#    and returns a set of weights. None of them can see the future.
# ---------------------------------------------------------------------------


def equal_weight_strategy(past: pd.DataFrame, risk_free_rate: float, max_weight: float) -> np.ndarray:
    """The simple benchmark: the same amount in every stock. No estimates needed."""
    n_stocks = past.shape[1]  # .shape[1] = the number of columns
    return np.full(n_stocks, 1 / n_stocks)


def min_volatility_strategy(past: pd.DataFrame, risk_free_rate: float, max_weight: float) -> np.ndarray:
    """Smoothest ride: the lowest-volatility mix (uses only the covariance matrix)."""
    covariance = past.cov() * TRADING_DAYS_PER_YEAR
    return min_variance_weights(covariance, max_weight)


def max_sharpe_strategy(past: pd.DataFrame, risk_free_rate: float, max_weight: float) -> np.ndarray:
    """Best return per unit of risk: the highest Sharpe ratio on past data."""
    expected_returns = past.mean() * TRADING_DAYS_PER_YEAR
    covariance = past.cov() * TRADING_DAYS_PER_YEAR
    return max_sharpe_weights(expected_returns, covariance, risk_free_rate, max_weight)


def max_sortino_strategy(past: pd.DataFrame, risk_free_rate: float, max_weight: float) -> np.ndarray:
    """Best return per unit of DOWNSIDE risk: the highest Sortino ratio on past data."""
    returns = past.values  # .values = the numbers only, as a numpy array (faster)

    def negative_sortino(weights):
        return -sortino_ratio(returns @ weights, risk_free_rate)

    return optimise_multistart(negative_sortino, returns.shape[1], max_weight)


def min_expected_shortfall_strategy(past: pd.DataFrame, risk_free_rate: float, max_weight: float) -> np.ndarray:
    """Smallest bad days: the lowest average loss on the worst 5% of past days."""
    returns = past.values

    def shortfall(weights):
        return expected_shortfall(returns @ weights)

    return optimise_multistart(shortfall, returns.shape[1], max_weight)


def min_drawdown_strategy(past: pd.DataFrame, risk_free_rate: float, max_weight: float) -> np.ndarray:
    """Shallowest falls: the smallest maximum drawdown over the past window."""
    returns = past.values

    def drawdown_size(weights):
        return -max_drawdown(returns @ weights)  # a positive number to minimise

    return optimise_multistart(drawdown_size, returns.shape[1], max_weight)


# Plain-English name -> strategy function. The app loops over this dictionary.
STRATEGIES = {
    "Equal weight": equal_weight_strategy,
    "Smoothest ride (min volatility)": min_volatility_strategy,
    "Best return per unit of risk (max Sharpe)": max_sharpe_strategy,
    "Best return per unit of downside risk (max Sortino)": max_sortino_strategy,
    "Smallest bad days (min Expected Shortfall)": min_expected_shortfall_strategy,
    "Shallowest falls (min drawdown)": min_drawdown_strategy,
}


# ---------------------------------------------------------------------------
# 4. The walk-forward engine
# ---------------------------------------------------------------------------


def rebalance_dates(returns: pd.DataFrame, lookback_years: int, rebalance_months: int) -> list:
    """The dates on which weights are chosen: after the first look-back window,
    then every rebalance_months months until the data runs out."""
    dates = []
    date = returns.index[0] + pd.DateOffset(years=lookback_years)
    while date < returns.index[-1]:
        dates.append(date)
        date = date + pd.DateOffset(months=rebalance_months)
    return dates


def walk_forward(
    returns: pd.DataFrame,
    strategy,
    lookback_years: int,
    rebalance_months: int,
    cost_rate: float,
    risk_free: float | pd.Series,
    max_weight: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run one strategy through time, using only past data at each decision.

    returns: daily returns, one column per stock.
    cost_rate: trading cost as a share of the amount traded, e.g. 0.001 = 0.1%.
    risk_free: a fixed rate, or a Series of daily rates (see average_risk_free_rate).

    Returns two things:
      wealth: the portfolio's value each day, starting at 100
      history: the weights chosen at each rebalance date, and the turnover
    """
    dates = rebalance_dates(returns, lookback_years, rebalance_months)
    n_stocks = returns.shape[1]

    value = 100.0
    current_weights = np.zeros(n_stocks)  # start in cash, so the first purchase is a trade
    wealth_pieces = []
    history_rows = []

    for i, start in enumerate(dates):
        # enumerate gives a counter (i) as well as each date.
        # The holding period runs from this rebalance date to the next one (or the end).
        if i + 1 < len(dates):
            end = dates[i + 1]
            holding = returns[(returns.index >= start) & (returns.index < end)]
        else:
            holding = returns[returns.index >= start]
        if holding.empty:
            break

        # 1. Choose weights using ONLY the look-back window before this date.
        window_start = start - pd.DateOffset(years=lookback_years)
        past = returns[(returns.index >= window_start) & (returns.index < start)]
        # The risk-free rate that applied during that same look-back window
        # (so no future information sneaks in).
        window_rate = average_risk_free_rate(risk_free, past.index[0], past.index[-1])
        target_weights = strategy(past, window_rate, max_weight)

        # 2. Pay trading costs on everything that changes.
        #    Turnover = the total of the weight changes (buys plus sells).
        turnover = np.abs(target_weights - current_weights).sum()
        value = value * (1 - turnover * cost_rate)

        # 3. Hold the weights until the next rebalance. Between rebalances the
        #    weights drift as prices move (no daily trading), as in real life.
        growth = (1 + holding).cumprod().values  # each stock's growth since the rebalance
        holdings = growth * target_weights  # value of each position, per 1 invested
        daily_values = value * holdings.sum(axis=1)
        wealth_pieces.append(pd.Series(daily_values, index=holding.index))

        # 4. Where the drifted weights ended up, ready for the next rebalance.
        current_weights = holdings[-1] / holdings[-1].sum()
        value = daily_values[-1]

        row = {"Rebalance date": start, "Turnover": turnover}
        for ticker, weight in zip(returns.columns, target_weights):
            row[ticker] = weight
        history_rows.append(row)

    # Add a starting point of 100 on the last day before the test begins,
    # so every line on the chart starts from the same place.
    day_before = returns.index[returns.index < dates[0]][-1]
    wealth = pd.concat([pd.Series([100.0], index=[day_before])] + wealth_pieces)

    history = pd.DataFrame(history_rows).set_index("Rebalance date")
    return wealth, history


def performance_summary(wealth: pd.Series, risk_free_rate: float) -> dict:
    """The key results of one strategy, from its daily wealth path."""
    daily = wealth.pct_change().dropna().values
    years = (wealth.index[-1] - wealth.index[0]).days / 365.25

    volatility = np.std(daily, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return {
        # The steady yearly growth rate that matches the start and end values.
        "Annual growth (compound)": (wealth.iloc[-1] / wealth.iloc[0]) ** (1 / years) - 1,
        "Volatility": volatility,
        "Sharpe ratio": (np.mean(daily) * TRADING_DAYS_PER_YEAR - risk_free_rate) / volatility,
        "Sortino ratio": sortino_ratio(daily, risk_free_rate),
        "Maximum drawdown": max_drawdown(daily),
        "Expected Shortfall (95%)": expected_shortfall(daily),
        "Final value (from 100)": wealth.iloc[-1],
    }


def run_all_strategies(
    returns: pd.DataFrame,
    lookback_years: int,
    rebalance_months: int,
    cost_rate: float,
    risk_free: float | pd.Series,
    max_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run every strategy in STRATEGIES through the same walk-forward test.

    risk_free: a fixed rate, or a Series of daily rates (see average_risk_free_rate).

    Returns:
      wealth_table: one column per strategy (daily values starting at 100)
      summary_table: one row per strategy (the performance_summary results)
      histories: {strategy name: its table of weights at each rebalance}
    """
    wealth_columns = {}
    summary_rows = {}
    histories = {}

    for name, strategy in STRATEGIES.items():
        wealth, history = walk_forward(
            returns, strategy, lookback_years, rebalance_months,
            cost_rate, risk_free, max_weight,
        )
        # Judge each strategy against the risk-free rate that applied during the test.
        test_rate = average_risk_free_rate(risk_free, wealth.index[0], wealth.index[-1])
        wealth_columns[name] = wealth
        summary_rows[name] = performance_summary(wealth, test_rate)
        summary_rows[name]["Average turnover per rebalance"] = history["Turnover"].mean()
        histories[name] = history

    wealth_table = pd.DataFrame(wealth_columns)
    # "orient='index'" makes each dictionary key (strategy name) a row.
    summary_table = pd.DataFrame.from_dict(summary_rows, orient="index")
    return wealth_table, summary_table, histories