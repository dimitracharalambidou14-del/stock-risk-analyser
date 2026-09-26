# montecarlo.py
# Monte Carlo simulation of a portfolio's possible futures: the "goal planner".
# It handles a starting amount, regular monthly contributions, fees and inflation.
# Like metrics.py and portfolio.py, there's no Streamlit code here:
# numbers go in, numbers come out.

import numpy as np
import pandas as pd
from scipy.stats import norm  # the normal (bell-curve) distribution

# Simulate the future month by month.
STEPS_PER_YEAR = 12

# The projection horizons offered in the app, in years.
HORIZON_YEARS = {"1Y": 1, "5Y": 5, "10Y": 10, "20Y": 20, "30Y": 30}

# The extra return assumed for each unit of risk in the long run (a long-run
# Sharpe ratio). 0.30 is deliberately cautious: for a typical diversified share
# portfolio with about 15% volatility, it means roughly 4.5% a year above cash.
LONG_RUN_SHARPE = 0.30


def long_run_expected_return(risk_free_rate: float, volatility: float) -> float:
    """A cautious long-run return assumption: cash rate + a premium for the risk taken.

        expected return = risk-free rate + LONG_RUN_SHARPE x volatility

    Riskier portfolios get a higher assumed return, but nothing like the 15-20% a
    year that a few years of strong stock performance can suggest.
    """
    return risk_free_rate + LONG_RUN_SHARPE * volatility


# ---------------------------------------------------------------------------
# 1. Two ways of generating random monthly GROWTH FACTORS
#    (a growth factor of 1.02 means +2% that month; 0.97 means -3%).
#    Both return a table: one ROW per simulated future, one COLUMN per month.
# ---------------------------------------------------------------------------


def normal_growth_factors(
    annual_return: float,
    annual_volatility: float,
    years: int,
    n_paths: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Method 1, the normal model (geometric Brownian motion).

    Each month's log-return is drawn from a normal distribution with:
        average   = (annual_return - 0.5 x annual_volatility^2) x dt
        std. dev. = annual_volatility x sqrt(dt)
    where dt = 1/12 (one month, measured in years).
    Writing 10_000 is the same as 10000: the underscore just makes it easier to read.
    """
    n_months = years * STEPS_PER_YEAR
    dt = 1 / STEPS_PER_YEAR

    # The "- 0.5 x volatility^2" is the volatility drag from Phase 4:
    # it turns the ARITHMETIC average return into the typical COMPOUND growth rate.
    drift = (annual_return - 0.5 * annual_volatility ** 2) * dt
    shock_size = annual_volatility * np.sqrt(dt)

    # A fixed seed gives the same "random" numbers every run (reproducible).
    rng = np.random.default_rng(seed)
    random_shocks = rng.standard_normal((n_paths, n_months))

    # e^(log-return) turns each log-return into a growth factor.
    return np.exp(drift + shock_size * random_shocks)


def monthly_portfolio_returns(daily_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """The portfolio's actual past MONTHLY returns, built from the stocks' daily returns.

    daily_returns: one column per stock (e.g. from daily_returns_table in portfolio.py).
    weights: the portfolio weights, in the same order as the columns.
    Assumes the weights are kept constant (rebalanced back to target each day).
    """
    # Each day's portfolio return = sum of (weight x that stock's return).
    daily_portfolio = daily_returns @ weights

    # Compound the days within each calendar month: (1 + r1) x (1 + r2) x ... - 1.
    # .resample("ME") groups the rows by month ("ME" = month end).
    monthly = (1 + daily_portfolio).resample("ME").prod() - 1

    # Drop the first and last months: they're only partly covered by the data,
    # so they'd look calmer than a full month really is.
    return monthly.iloc[1:-1]


def bootstrap_growth_factors(
    monthly_returns: pd.Series,
    years: int,
    n_paths: int = 10_000,
    seed: int = 42,
    target_annual_return: float | None = None,
) -> np.ndarray:
    """Method 2, the historical bootstrap.

    Each simulated month copies one actual past month, picked at random "with
    replacement" (the same month can be picked again and again). No bell curve is
    assumed, so any fat tails in the real data carry through.

    target_annual_return: if given, the past months are first shifted ("re-centred")
    so their AVERAGE matches this return, while keeping their real ups and downs.
    That way both methods use the same return assumption, and only the SHAPE differs.
    """
    past = monthly_returns.values
    if target_annual_return is not None:
        past = past - past.mean() + target_annual_return / STEPS_PER_YEAR

    n_months = years * STEPS_PER_YEAR
    rng = np.random.default_rng(seed)
    sampled_returns = rng.choice(past, size=(n_paths, n_months), replace=True)
    return 1 + sampled_returns


# ---------------------------------------------------------------------------
# 2. Turning growth factors into portfolio values, with contributions,
#    fees and (optionally) inflation
# ---------------------------------------------------------------------------


def build_paths(
    growth_factors: np.ndarray,
    start_value: float,
    monthly_contribution: float = 0.0,
    annual_fee: float = 0.0,
    inflation: float = 0.0,
    todays_money: bool = False,
) -> np.ndarray:
    """Portfolio values month by month, for every simulated future.

    Each month: add the contribution, apply that month's growth, then take the fee.
        value next month = (value + contribution) x growth factor x fee factor

    The monthly contribution is assumed to RISE WITH INFLATION, so it keeps the
    same buying power. That makes "today's money" simple: dividing by inflation
    turns the rising contribution back into a constant amount, and turns
    nominal growth into REAL growth (growth after inflation).

    Returns a table: one row per path, one column per month (column 0 = start).
    """
    n_paths, n_months = growth_factors.shape  # .shape gives (rows, columns)

    # Convert yearly rates into the equivalent monthly factors,
    # e.g. a 0.5% yearly fee -> multiply by about 0.99958 each month.
    monthly_fee_factor = (1 - annual_fee) ** (1 / STEPS_PER_YEAR)
    monthly_inflation_factor = (1 + inflation) ** (1 / STEPS_PER_YEAR)

    # np.empty makes a table of the right size, ready to be filled in.
    values = np.empty((n_paths, n_months + 1))
    values[:, 0] = start_value  # every path starts at the starting amount

    # range(n_months) counts 0, 1, 2, ... up to n_months - 1.
    for month in range(n_months):
        if todays_money:
            # In today's money: a constant contribution, and growth after inflation.
            contribution = monthly_contribution
            growth = growth_factors[:, month] / monthly_inflation_factor
        else:
            # In future pounds: the contribution rises with inflation each month.
            contribution = monthly_contribution * monthly_inflation_factor ** month
            growth = growth_factors[:, month]

        # All 10,000 paths are updated at once (a whole column at a time).
        values[:, month + 1] = (values[:, month] + contribution) * growth * monthly_fee_factor

    return values


def total_paid_in(
    start_value: float,
    monthly_contribution: float,
    years: int,
    inflation: float = 0.0,
    todays_money: bool = False,
) -> np.ndarray:
    """The running total paid in, month by month, in the same units as build_paths."""
    n_months = years * STEPS_PER_YEAR

    if todays_money:
        # A constant contribution each month.
        contributions = np.full(n_months, monthly_contribution)
    else:
        # A contribution that rises with inflation each month.
        monthly_inflation_factor = (1 + inflation) ** (1 / STEPS_PER_YEAR)
        contributions = monthly_contribution * monthly_inflation_factor ** np.arange(n_months)

    # np.cumsum gives the running total; the 0 in front is "month 0, nothing added yet".
    return start_value + np.concatenate([[0.0], np.cumsum(contributions)])


# ---------------------------------------------------------------------------
# 3. Summarising the simulated futures
# ---------------------------------------------------------------------------


def percentile_bands(paths: np.ndarray, years: int) -> pd.DataFrame:
    """For every month, the 5th, 50th and 95th percentile of the simulated values.

    Returns a table: one row per month (labelled by time in years),
    with columns "5th", "50th" and "95th".
    """
    # axis=0 means "work down each column", i.e. across all paths for each month.
    fifth, median, ninety_fifth = np.percentile(paths, [5, 50, 95], axis=0)

    # The time of each column, in years: 0, 1/12, 2/12, ... up to the horizon.
    time_in_years = np.linspace(0, years, paths.shape[1])

    return pd.DataFrame(
        {"5th": fifth, "50th": median, "95th": ninety_fifth},
        index=time_in_years,
    )


def final_value_summary(paths: np.ndarray, amount_paid_in: float, goal: float | None = None) -> dict:
    """Key facts about the simulated values at the END of the horizon.

    amount_paid_in: the total paid in by the end (to measure the chance of a loss).
    goal: a target amount, or None if there isn't one.
    """
    final_values = paths[:, -1]  # every row (:), last column (-1)

    summary = {
        "5th": np.percentile(final_values, 5),
        "50th": np.percentile(final_values, 50),
        "95th": np.percentile(final_values, 95),
        # (final_values < amount_paid_in) is True/False for each path;
        # the average of True/False values is the share that were True.
        "chance_of_loss": np.mean(final_values < amount_paid_in),
    }

    if goal is not None:
        summary["chance_of_goal"] = np.mean(final_values >= goal)

    return summary


def exact_percentile(
    start_value: float,
    annual_return: float,
    annual_volatility: float,
    years: int,
    percentile: float,
) -> float:
    """The exact answer from the formula, with no simulation. Used to CHECK the simulation.

    It only applies to a single lump sum with no contributions, fees or inflation
    adjustment. Under the normal model, log(final value / start value) is normal with
        average   = (annual_return - 0.5 x annual_volatility^2) x years
        std. dev. = annual_volatility x sqrt(years)
    """
    z = norm.ppf(percentile / 100)  # e.g. 50th percentile -> z = 0
    log_growth = (annual_return - 0.5 * annual_volatility ** 2) * years + z * annual_volatility * np.sqrt(years)
    return start_value * np.exp(log_growth)