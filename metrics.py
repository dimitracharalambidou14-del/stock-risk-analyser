# metrics.py
# All the risk and return calculations live in this file.
# Every function takes pandas data in and gives numbers out, with no
# Streamlit code, so the maths is easy to check, test and explain.

import numpy as np
import pandas as pd
from scipy.stats import norm  # the normal (bell-curve) distribution

# Roughly 252 trading days in a year (365 days minus weekends and public holidays).
TRADING_DAYS_PER_YEAR = 252

# Look-back periods used for the "returns over each period" table
# and for the analysis window.
RETURN_PERIODS = {
    "1W": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "1Y": pd.DateOffset(years=1),
    "5Y": pd.DateOffset(years=5),
}


def trailing_window(prices: pd.Series, lookback: pd.DateOffset) -> pd.Series:
    """Keep only the most recent part of a price series, e.g. the last 1 year."""
    start_date = prices.index[-1] - lookback
    return prices[prices.index >= start_date]


def daily_returns(prices: pd.Series) -> pd.Series:
    """Daily percentage changes, as decimals (0.01 means +1%).

    Return on day t = price on day t / price on day t-1 - 1
    """
    # .pct_change() applies the formula above to every day at once.
    # The first day has no "yesterday", so its value is blank; .dropna() removes it.
    return prices.pct_change().dropna()


def period_returns(prices: pd.Series) -> dict[str, float | None]:
    """Total return over the last day, week, month, year and 5 years.

    Returns a dictionary such as {"1D": 0.012, "1W": -0.03, ...}.
    "float | None" means each value is either a number or None;
    None means there isn't enough history for that period.
    """
    results = {}

    # 1 day: the latest price compared with the one before it.
    results["1D"] = prices.iloc[-1] / prices.iloc[-2] - 1

    # .items() gives each (label, look-back) pair from the dictionary in turn.
    for label, lookback in RETURN_PERIODS.items():
        window = trailing_window(prices, lookback)
        wanted_start = prices.index[-1] - lookback

        # If the data starts more than a week after the date we wanted,
        # the stock hasn't been listed long enough for this period.
        if window.index[0] - wanted_start > pd.Timedelta(days=7):
            results[label] = None
        else:
            # Return over the period = last price / first price - 1
            results[label] = window.iloc[-1] / window.iloc[0] - 1

    return results


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised volatility = standard deviation of daily returns x sqrt(252).

    Why the square root? If each day's return is independent with the same
    variance, variances ADD UP over time: a year's variance is about 252 times
    the daily variance. Volatility is the square root of variance, so it
    grows with the SQUARE ROOT of time.
    """
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def annualised_return(returns: pd.Series) -> float:
    """Average daily return x 252: the (arithmetic) average return per year.

    Averages grow in line with time, so we multiply by 252 (not its square root).
    """
    return returns.mean() * TRADING_DAYS_PER_YEAR


def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """Sharpe ratio = (annualised return - risk-free rate) / annualised volatility.

    risk_free_rate: a yearly rate as a decimal, e.g. 0.04 for 4%.
    It measures the extra return earned for each unit of risk taken.
    """
    excess_return = annualised_return(returns) - risk_free_rate
    return excess_return / annualised_volatility(returns)


def drawdown_series(prices: pd.Series) -> pd.Series:
    """How far below its highest price so far the stock is on each day.

    Drawdown on day t = price on day t / highest price up to day t - 1
    It is 0 on days that set a new high, and negative otherwise.
    """
    # .cummax() = "cumulative maximum": the highest value seen so far, day by day.
    running_peak = prices.cummax()
    return prices / running_peak - 1


def max_drawdown(prices: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """The largest peak-to-trough fall, plus the dates of that peak and trough.

    Returns three things at once (a "tuple"): (drawdown, peak_date, trough_date).
    """
    drawdowns = drawdown_series(prices)

    # .idxmin() gives the DATE of the lowest value: the bottom of the worst fall.
    trough_date = drawdowns.idxmin()

    # The peak is the highest price on or before that trough.
    # .loc[:trough_date] keeps everything up to and including the trough date.
    peak_date = prices.loc[:trough_date].idxmax()

    return drawdowns.min(), peak_date, trough_date


def aligned_returns(stock_prices: pd.Series, market_prices: pd.Series) -> pd.DataFrame:
    """Daily returns for a stock and a market index, on matching dates.

    Returns a table with two columns: "stock" and "market".
    """
    # Put both price series side by side, lined up by date.
    # The dictionary keys become the column names.
    prices = pd.concat({"stock": stock_prices, "market": market_prices}, axis=1)

    # Same approach as the comparison chart: forward-fill holiday gaps,
    # then drop dates before both series have started.
    prices = prices.ffill().dropna()

    return prices.pct_change().dropna()


def beta(returns: pd.DataFrame) -> float:
    """Beta = covariance(stock, market) / variance(market).

    returns: the output of aligned_returns (columns "stock" and "market").
    It's also the slope of the best-fit line when stock returns are plotted
    against market returns.
    """
    covariance = returns["stock"].cov(returns["market"])
    market_variance = returns["market"].var()
    return covariance / market_variance


def correlation(returns: pd.DataFrame) -> float:
    """Correlation between the stock's and the market's daily returns (-1 to +1)."""
    return returns["stock"].corr(returns["market"])


# ---------------------------------------------------------------------------
# Value at Risk (VaR) and Expected Shortfall (ES).
# All four functions return a POSITIVE number meaning a LOSS,
# e.g. 0.025 means "a loss of 2.5%" over one day.
# confidence=0.95 means "95% confidence"; the worst 5% of days are the "tail".
# ---------------------------------------------------------------------------


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: the loss NOT exceeded on (confidence x 100)% of past days.

    It's minus the 5th percentile of past daily returns (for 95% confidence).
    No assumption is made about the shape of the distribution.
    """
    # .quantile(0.05) sorts the returns and finds the value with 5% of days below it.
    return -returns.quantile(1 - confidence)


def historical_es(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Expected Shortfall: the AVERAGE loss on the worst 5% of past days."""
    cutoff = returns.quantile(1 - confidence)
    worst_days = returns[returns <= cutoff]  # keep only the days in the tail
    return -worst_days.mean()


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric VaR, assuming daily returns follow a NORMAL distribution.

    VaR = -(average + z x standard deviation), where z is the point on the
    standard normal curve with 5% below it (z is about -1.645 at 95%).
    """
    # norm.ppf is the inverse of the normal distribution: probability -> z value.
    z = norm.ppf(1 - confidence)
    return -(returns.mean() + z * returns.std())


def parametric_es(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric Expected Shortfall, assuming a NORMAL distribution.

    ES = -(average - standard deviation x pdf(z) / tail probability),
    where pdf(z) is the height of the normal curve at z.
    At 95% confidence this works out as about 2.063 x the standard deviation.
    """
    tail_probability = 1 - confidence  # e.g. 0.05
    z = norm.ppf(tail_probability)
    return -(returns.mean() - returns.std() * norm.pdf(z) / tail_probability)


def excess_kurtosis(returns: pd.Series) -> float:
    """How "fat-tailed" the returns are. 0 for a normal distribution;
    higher values mean extreme days happen more often than a normal curve predicts."""
    # pandas' .kurt() already subtracts 3, giving EXCESS kurtosis.
    return returns.kurt()