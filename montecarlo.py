# montecarlo.py
# Monte Carlo simulation of a portfolio's possible future values.
# Like metrics.py and portfolio.py, there's no Streamlit code here:
# numbers go in, numbers come out.

import numpy as np
import pandas as pd
from scipy.stats import norm  # the normal (bell-curve) distribution

# Simulate the future month by month.
STEPS_PER_YEAR = 12

# The projection horizons offered in the app, in years.
HORIZON_YEARS = {"1Y": 1, "5Y": 5, "10Y": 10}


def simulate_paths(
    start_value: float,
    annual_return: float,
    annual_volatility: float,
    years: int,
    n_paths: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate many possible futures for a portfolio (geometric Brownian motion).

    Each month, the value is multiplied by e^(log-return), where the log-return
    is drawn at random from a normal distribution with:
        average   = (annual_return - 0.5 x annual_volatility^2) x dt
        std. dev. = annual_volatility x sqrt(dt)
    and dt = 1/12 (one month, measured in years).

    Returns a 2-D array: one ROW per simulated path, one COLUMN per month
    (column 0 is today, so every path starts at start_value).
    Writing 10_000 is the same as 10000: the underscore just makes it easier to read.
    """
    n_steps = years * STEPS_PER_YEAR
    dt = 1 / STEPS_PER_YEAR

    # The "- 0.5 x volatility^2" is the volatility drag from Phase 4:
    # it turns the ARITHMETIC average return into the typical COMPOUND growth rate.
    drift = (annual_return - 0.5 * annual_volatility ** 2) * dt
    shock_size = annual_volatility * np.sqrt(dt)

    # A random-number generator with a fixed seed gives the same "random"
    # numbers every run, so results can be reproduced and checked.
    rng = np.random.default_rng(seed)

    # A table of standard normal random numbers: n_paths rows x n_steps columns.
    random_shocks = rng.standard_normal((n_paths, n_steps))

    # Each month's log-return = drift + shock_size x (a random normal number).
    log_returns = drift + shock_size * random_shocks

    # Adding up log-returns along each row (axis=1) gives the total log-growth so far.
    cumulative_log_growth = np.cumsum(log_returns, axis=1)

    # Put a column of zeros in front, so month 0 has "no growth yet".
    # np.hstack joins arrays side by side.
    cumulative_log_growth = np.hstack([np.zeros((n_paths, 1)), cumulative_log_growth])

    # e^(total log-growth) turns it back into a growth factor, e.g. 1.25 = +25%.
    return start_value * np.exp(cumulative_log_growth)


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


def final_value_summary(paths: np.ndarray, start_value: float) -> dict:
    """Key facts about the simulated values at the END of the horizon."""
    final_values = paths[:, -1]  # every row (:), last column (-1)

    return {
        "5th": np.percentile(final_values, 5),
        "50th": np.percentile(final_values, 50),
        "95th": np.percentile(final_values, 95),
        # (final_values < start_value) is True/False for each path;
        # the average of True/False values is the share that were True.
        "chance_of_loss": np.mean(final_values < start_value),
    }


def exact_percentile(
    start_value: float,
    annual_return: float,
    annual_volatility: float,
    years: int,
    percentile: float,
) -> float:
    """The exact answer from the formula, with no simulation. Used to CHECK the simulation.

    Under this model, log(final value / start value) is normally distributed with
        average   = (annual_return - 0.5 x annual_volatility^2) x years
        std. dev. = annual_volatility x sqrt(years)
    """
    z = norm.ppf(percentile / 100)  # e.g. 50th percentile -> z = 0
    log_growth = (annual_return - 0.5 * annual_volatility ** 2) * years + z * annual_volatility * np.sqrt(years)
    return start_value * np.exp(log_growth)


# ---------------------------------------------------------------------------
# A SECOND METHOD: the historical bootstrap.
# Instead of assuming a normal distribution, re-use the portfolio's REAL past
# monthly returns, picked at random. Comparing the two methods shows how much
# the answer depends on the model chosen ("model risk").
# ---------------------------------------------------------------------------


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


def bootstrap_paths(
    start_value: float,
    monthly_returns: pd.Series,
    years: int,
    n_paths: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate futures by re-using real past monthly returns, chosen at random.

    Each simulated month copies one actual past month, picked "with replacement"
    (the same month can be picked again and again). No bell curve is assumed, so any
    fat tails in the real data carry through into the simulation.

    Returns the same shape of array as simulate_paths: one row per path,
    one column per month, with column 0 = start_value.
    """
    n_steps = years * STEPS_PER_YEAR
    rng = np.random.default_rng(seed)

    # rng.choice picks values at random from the list of past monthly returns,
    # filling a table of n_paths rows x n_steps columns.
    sampled_returns = rng.choice(monthly_returns.values, size=(n_paths, n_steps), replace=True)

    # Multiply the growth factors (1 + return) together along each row.
    # np.cumprod = "cumulative product": the running total when multiplying.
    growth = np.cumprod(1 + sampled_returns, axis=1)

    # Put a column of 1s in front, so month 0 has "no growth yet".
    growth = np.hstack([np.ones((n_paths, 1)), growth])

    return start_value * growth