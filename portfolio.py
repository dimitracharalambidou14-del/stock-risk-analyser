# portfolio.py
# Portfolio maths: estimating the inputs, measuring a portfolio's return and
# risk, and finding "optimal" weights. Like metrics.py, no Streamlit code here.

import numpy as np
import pandas as pd
from scipy.optimize import minimize  # scipy's general-purpose optimiser

from metrics import TRADING_DAYS_PER_YEAR

# Risk-aversion coefficient "A" for each risk appetite.
# Higher A = risk is penalised more heavily = a more cautious portfolio.
# These values are judgement calls (typical estimates range from about 1 to 10).
RISK_AVERSION = {"Low": 10.0, "Medium": 4.0, "High": 1.5}


def daily_returns_table(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily returns for every stock at once (one column per stock)."""
    return closes.pct_change().dropna()


def estimate_inputs(closes: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate the two inputs to mean-variance optimisation from past prices.

    closes: daily closing prices, one column per stock, lined up by date.

    Returns two things:
      expected_returns: each stock's average return per year (annualised x 252)
      covariance: the annualised covariance matrix (a stock-by-stock table)
    """
    returns = daily_returns_table(closes)

    # Averages scale with time, so multiply the daily average by 252.
    expected_returns = returns.mean() * TRADING_DAYS_PER_YEAR

    # Variances and covariances ALSO scale with time (that's why volatility,
    # their square root, scales with sqrt(252)), so multiply by 252.
    covariance = returns.cov() * TRADING_DAYS_PER_YEAR

    return expected_returns, covariance


def correlation_matrix(closes: pd.DataFrame) -> pd.DataFrame:
    """Correlation between every pair of stocks' daily returns (-1 to +1)."""
    return daily_returns_table(closes).corr()


def individual_volatilities(covariance: pd.DataFrame) -> pd.Series:
    """Each stock's own annual volatility.

    The DIAGONAL of the covariance matrix (top-left to bottom-right) holds each
    stock's variance with itself; the square root of a variance is a volatility.
    """
    return pd.Series(np.sqrt(np.diag(covariance)), index=covariance.index)


def portfolio_return(weights: np.ndarray, expected_returns: pd.Series) -> float:
    """Expected portfolio return = sum of (weight x expected return).

    In matrix notation: w · mu. The @ symbol means "matrix multiply", which here
    multiplies each weight by its stock's return and adds them all up.
    """
    return float(weights @ expected_returns.values)


def portfolio_volatility(weights: np.ndarray, covariance: pd.DataFrame) -> float:
    """Portfolio volatility = square root of (w^T x Covariance x w).

    This adds up every pair of stocks: weight_i x weight_j x covariance_ij.
    Pairs that don't move together partly cancel out, which is diversification.
    """
    variance = weights @ covariance.values @ weights
    return float(np.sqrt(variance))


def portfolio_sharpe(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """Sharpe ratio of a portfolio: (return - risk-free rate) / volatility."""
    excess = portfolio_return(weights, expected_returns) - risk_free_rate
    return excess / portfolio_volatility(weights, covariance)


# ---------------------------------------------------------------------------
# OPTIMISATION
# ---------------------------------------------------------------------------


def weights_sum_minus_one(weights: np.ndarray) -> float:
    """Used as a constraint: the optimiser must keep this equal to 0,
    which means the weights must add up to exactly 1 (100% invested)."""
    return np.sum(weights) - 1


def optimise(
    objective,
    n_stocks: int,
    max_weight: float,
    extra_constraints: list | None = None,
    starting_guess: np.ndarray | None = None,
) -> np.ndarray:
    """Find the weights that make objective(weights) as SMALL as possible.

    Rules the answer must follow (the "constraints"):
      - every weight is between 0 and max_weight (no short-selling, no borrowing)
      - the weights add up to 1 (fully invested)
      - plus any extra_constraints passed in (e.g. "hit this target return")

    starting_guess: where the search begins. If none is given, start from equal weights.
    """
    if starting_guess is None:
        starting_guess = np.full(n_stocks, 1 / n_stocks)

    # One (lowest, highest) pair per stock. [x] * 3 makes the list [x, x, x].
    bounds = [(0.0, max_weight)] * n_stocks

    # "eq" = equality constraint: weights_sum_minus_one(weights) must equal 0.
    constraints = [{"type": "eq", "fun": weights_sum_minus_one}]
    if extra_constraints:
        # Adding two lists together joins them into one longer list.
        constraints = constraints + extra_constraints

    # SLSQP is an optimisation method that can handle bounds and constraints.
    result = minimize(
        objective,
        starting_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"The optimiser could not find a solution: {result.message}")

    # Tidy up tiny rounding errors (e.g. -0.0000000001 instead of 0),
    # then rescale so the weights add up to exactly 1.
    weights = np.clip(result.x, 0, None)  # replace any negative value with 0
    return weights / weights.sum()


def min_variance_weights(covariance: pd.DataFrame, max_weight: float = 1.0) -> np.ndarray:
    """The lowest-risk portfolio. Note: it doesn't use expected returns at all."""

    # A small function defined INSIDE another one can use its variables
    # (here, covariance). The optimiser calls it with different weights.
    def variance(weights):
        return weights @ covariance.values @ weights

    return optimise(variance, len(covariance), max_weight)


def max_sharpe_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float = 1.0,
) -> np.ndarray:
    """The portfolio with the highest Sharpe ratio (best return per unit of risk)."""

    # The optimiser MINIMISES, so to find the HIGHEST Sharpe ratio
    # we ask it to find the LOWEST negative Sharpe ratio.
    def negative_sharpe(weights):
        return -portfolio_sharpe(weights, expected_returns, covariance, risk_free_rate)

    return optimise(negative_sharpe, len(expected_returns), max_weight)


def max_return_weights(expected_returns: pd.Series, max_weight: float = 1.0) -> np.ndarray:
    """The portfolio with the highest expected return allowed by the cap.
    It's the top end of the efficient frontier."""

    def negative_return(weights):
        return -portfolio_return(weights, expected_returns)

    return optimise(negative_return, len(expected_returns), max_weight)


def risk_appetite_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
    max_weight: float = 1.0,
) -> np.ndarray:
    """The portfolio with the best mean-variance "score" for a given risk aversion:

        score = expected return - 0.5 x risk_aversion x variance

    A cautious investor (high risk_aversion) is penalised heavily for variance.
    """

    def negative_score(weights):
        expected = portfolio_return(weights, expected_returns)
        variance = weights @ covariance.values @ weights
        return -(expected - 0.5 * risk_aversion * variance)

    return optimise(negative_score, len(expected_returns), max_weight)


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    max_weight: float = 1.0,
    n_points: int = 40,
) -> pd.DataFrame:
    """Trace the efficient frontier: the lowest-risk portfolio for each target return.

    It runs from the minimum-variance portfolio (lowest risk) up to the
    highest-return portfolio. Returns a table with columns
    "Volatility" and "Expected return", one row per point on the curve.
    """
    n_stocks = len(expected_returns)

    # The two ends of the frontier.
    lowest_risk = min_variance_weights(covariance, max_weight)
    highest_return = max_return_weights(expected_returns, max_weight)
    start = portfolio_return(lowest_risk, expected_returns)
    end = portfolio_return(highest_return, expected_returns)

    def variance(weights):
        return weights @ covariance.values @ weights

    rows = []

    # np.linspace(start, end, 40) gives 40 evenly spaced target returns.
    for target in np.linspace(start, end, n_points):

        # Extra rule: the portfolio's return must equal this target.
        def return_gap(weights):
            return portfolio_return(weights, expected_returns) - target

        try:
            weights = optimise(
                variance,
                n_stocks,
                max_weight,
                extra_constraints=[{"type": "eq", "fun": return_gap}],
            )
        except RuntimeError:
            continue  # "continue" skips to the next target if this one fails

        rows.append(
            {
                "Volatility": portfolio_volatility(weights, covariance),
                "Expected return": portfolio_return(weights, expected_returns),
            }
        )

    return pd.DataFrame(rows)


def random_portfolios(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    n_portfolios: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Thousands of random portfolios, to show the "cloud" of possible mixes.

    rng.dirichlet makes random weights that are all positive and add up to 1.
    A fixed "seed" makes the random numbers the same on every run (reproducible).
    """
    rng = np.random.default_rng(seed)
    all_weights = rng.dirichlet(np.ones(len(expected_returns)), size=n_portfolios)

    rows = []
    for weights in all_weights:
        rows.append(
            {
                "Volatility": portfolio_volatility(weights, covariance),
                "Expected return": portfolio_return(weights, expected_returns),
                "Sharpe ratio": portfolio_sharpe(
                    weights, expected_returns, covariance, risk_free_rate
                ),
            }
        )

    return pd.DataFrame(rows)


def summary_table(
    portfolios: dict[str, np.ndarray],
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
) -> pd.DataFrame:
    """A table comparing several portfolios: their weights, return, volatility and Sharpe.

    portfolios: a dictionary such as {"Equal weight": weights, "Minimum variance": weights}
    """
    rows = []

    for name, weights in portfolios.items():
        row = {"Portfolio": name}

        # zip() pairs each ticker with its weight, so we get one column per stock.
        for ticker, weight in zip(expected_returns.index, weights):
            row[ticker] = weight

        row["Expected return"] = portfolio_return(weights, expected_returns)
        row["Volatility"] = portfolio_volatility(weights, covariance)
        row["Sharpe ratio"] = portfolio_sharpe(weights, expected_returns, covariance, risk_free_rate)
        rows.append(row)

    # A list of dictionaries becomes a table: one dictionary per row.
    # .set_index("Portfolio") uses the portfolio names as the row labels.
    return pd.DataFrame(rows).set_index("Portfolio")