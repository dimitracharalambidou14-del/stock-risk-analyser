# summary.py
# Turns the risk numbers into plain-English sentences using simple rules.
# No AI: just if/else rules with clear thresholds, so every sentence can be
# traced back to a number and a rule.
# The wording deliberately describes the PAST and never says "buy" or "sell".

# ---------------------------------------------------------------------------
# THRESHOLDS ("bands"). Each band is a pair: (upper limit, description).
# These are rules of thumb, i.e. judgement calls. Keeping them together at
# the top makes them easy to find, explain and change.
# ---------------------------------------------------------------------------
VOLATILITY_BANDS = [
    (0.15, "low (similar to a broad market index)"),
    (0.25, "moderate"),
    (0.40, "high"),
]  # 40% or more: "very high"

DRAWDOWN_BANDS = [
    (0.10, "a shallow"),
    (0.25, "a significant"),
    (0.50, "a deep"),
]  # 50% or more: "a severe"

SHARPE_BANDS = [
    (0.5, "a low"),
    (1.0, "a moderate"),
]  # 1.0 or more: "a strong"

BETA_BANDS = [
    (0.8, "has tended to move less than"),
    (1.2, "has tended to move roughly in line with"),
]  # 1.2 or more: "has tended to amplify the moves of"

# If the market explains less than this share of the stock's daily moves (R²),
# the relationship is too weak for beta to be meaningful.
WEAK_RELATIONSHIP_R2 = 0.05

# Readable names for the analysis window.
WINDOW_NAMES = {"1Y": "1 year", "5Y": "5 years"}


def label_from_bands(value: float, bands: list, above_all: str) -> str:
    """Return the description of the first band whose upper limit is above the value.

    Example: with VOLATILITY_BANDS, 0.20 -> "moderate" (it's below 0.25 but not 0.15).
    If the value is above every limit, return the "above_all" description.
    """
    for upper_limit, description in bands:
        if value < upper_limit:
            return description
    return above_all


# IMPORTANT RULE used below: we always classify the number AS IT IS DISPLAYED
# (rounded), so the words can never contradict the number the reader sees.
# round(x, 2) rounds x to 2 decimal places, e.g. round(0.4996, 2) -> 0.5.


def describe_returns(ticker: str, return_1y: float | None, return_5y: float | None) -> str:
    """Sentence about past returns over 1 and 5 years."""
    sentence = ""

    if return_1y is not None:
        direction = "gained" if return_1y >= 0 else "lost"
        sentence += (
            f"Over the past year, {ticker} {direction} {abs(return_1y):.1%} "
            "in total, including dividends."
        )

    if return_5y is not None:
        # Compound annual growth rate: the steady yearly rate that would turn
        # 1 into (1 + total return) over 5 years, i.e. (1 + R)^(1/5) - 1.
        # It's the same compounding used with interest rates.
        per_year = (1 + return_5y) ** (1 / 5) - 1
        sentence += (
            f" Over five years the total return was {return_5y:+.0%}, "
            f"equivalent to about {per_year:+.1%} a year compounded."
        )

    return sentence.strip()  # .strip() removes any spare space at the start


def describe_volatility(annual_vol: float, market_vol: float | None, benchmark_name: str) -> str:
    """Sentence about how bumpy the ride has been, compared with the chosen index."""
    shown_vol = round(annual_vol, 3)  # as displayed, e.g. 0.2071 -> 0.207 -> "20.7%"
    level = label_from_bands(shown_vol, VOLATILITY_BANDS, "very high")
    sentence = f"Its annualised volatility of {shown_vol:.1%} is {level}"

    if market_vol:  # only if we have the index's volatility to compare with
        ratio = annual_vol / market_vol
        # Name the index, so the reader knows exactly what it's compared with.
        sentence += (
            f": about {ratio:.1f} times as volatile as the {benchmark_name}, "
            f"which had a volatility of {market_vol:.1%} over the same period"
        )

    return sentence + "."


def describe_drawdown(max_dd: float, peak_date, trough_date, current_dd: float) -> str:
    """Sentence about the worst fall, what it takes to recover, and where it is now."""
    shown_dd = round(max_dd, 2)  # as displayed, e.g. -0.343 -> -0.34 -> "34%"

    if shown_dd == 0:
        return "It had no meaningful fall from a peak during this period."

    size = label_from_bands(abs(shown_dd), DRAWDOWN_BANDS, "a severe")

    # After a fall of D, you need a gain of 1 / (1 - D) - 1 to get back to the peak.
    # (max_dd is negative, so 1 + max_dd is the same as 1 - D.)
    gain_needed = 1 / (1 + max_dd) - 1

    sentence = (
        f"Its worst fall was {size} {abs(shown_dd):.0%}, from a peak on "
        f"{peak_date:%d %b %Y} to a low on {trough_date:%d %b %Y}. "
        f"Recovering from a fall of that size needs a gain of about {gain_needed:.0%}."
    )

    if current_dd > -0.01:
        sentence += " It is currently at or close to its highest level in this period."
    else:
        sentence += f" It is currently {abs(current_dd):.0%} below its peak."

    return sentence


def describe_sharpe(sharpe: float, risk_free_rate: float) -> str:
    """Sentence about return per unit of risk."""
    shown_sharpe = round(sharpe, 2)  # as displayed, e.g. 0.4996 -> 0.5 -> "0.50"

    if shown_sharpe < 0:
        return (
            f"Its Sharpe ratio of {shown_sharpe:.2f} is negative: on average it returned "
            f"less than the {risk_free_rate:.1%} risk-free rate, so the extra risk "
            "was not rewarded over this period."
        )

    quality = label_from_bands(shown_sharpe, SHARPE_BANDS, "a strong")
    return (
        f"Its Sharpe ratio of {shown_sharpe:.2f} points to {quality} return for the "
        f"risk taken over this period, compared with a {risk_free_rate:.1%} risk-free rate."
    )


def describe_beta(beta: float, r_squared: float, benchmark_name: str) -> str:
    """Sentence about sensitivity to the market and how much of the risk is market-driven."""
    shown_beta = round(beta, 2)

    # First check: is there a real relationship at all? If the market explains
    # almost none of the stock's moves, beta is just noise, so say that instead.
    if r_squared < WEAK_RELATIONSHIP_R2:
        return (
            f"Its daily moves showed almost no relationship with the {benchmark_name}: "
            f"less than {WEAK_RELATIONSHIP_R2:.0%} of them were explained by it. "
            f"So its beta of {shown_beta:.2f} isn't a meaningful measure against this index, "
            "and the stock's home-market index is likely to be a better comparison."
        )

    if shown_beta < 0:
        behaviour = "has tended to move in the opposite direction to"
    else:
        behaviour = label_from_bands(shown_beta, BETA_BANDS, "has tended to amplify the moves of")

    sentence = (
        f"With a beta of {shown_beta:.2f}, it {behaviour} the {benchmark_name}: on past data, "
        f"a 1% market move went with about a {shown_beta:.2f}% move in the stock, on average."
    )

    if r_squared < 0.3:
        sentence += (
            f" But only {r_squared:.0%} of its daily movements were explained by the "
            "market, so most of its risk was specific to the company."
        )
    elif r_squared > 0.6:
        sentence += (
            f" {r_squared:.0%} of its daily movements were explained by the market, "
            "so market-wide swings were the main driver."
        )

    return sentence


def describe_tail_risk(
    hist_var: float, hist_es: float, param_es: float, kurtosis: float, confidence: float
) -> str:
    """Sentence about bad days: how often, how big, and whether tails are fat."""
    # 95% confidence -> the worst 5% of days -> about 1 day in 20.
    one_in = round(1 / (1 - confidence))

    sentence = (
        f"Based on past daily returns, a loss of more than {hist_var:.1%} happened on "
        f"about 1 trading day in {one_in} (around {10000 * hist_var:,.0f} on 10,000 invested), "
        f"and on those bad days the average loss was {hist_es:.1%}."
    )

    if kurtosis > 1 and hist_es > param_es:
        sentence += (
            " Extreme days were more common than a normal bell curve predicts "
            "(fat tails), so normal-distribution estimates understate the risk of very bad days."
        )

    return sentence


def build_summary(ticker: str, stats: dict) -> list[str]:
    """Build the full plain-English summary as a list of sentences.

    stats: a dictionary of the calculated numbers (see app.py for the keys).
    Optional values (such as beta) may be None if they couldn't be calculated.
    """
    sentences = []

    returns_sentence = describe_returns(ticker, stats["return_1y"], stats["return_5y"])
    if returns_sentence:  # skip it if it came back empty
        sentences.append(returns_sentence)

    sentences.append(
        describe_volatility(stats["annual_vol"], stats["market_vol"], stats["benchmark_name"])
    )

    sentences.append(
        describe_drawdown(
            stats["max_drawdown"],
            stats["peak_date"],
            stats["trough_date"],
            stats["current_drawdown"],
        )
    )

    sentences.append(describe_sharpe(stats["sharpe"], stats["risk_free_rate"]))

    if stats["beta"] is not None:
        sentences.append(
            describe_beta(stats["beta"], stats["r_squared"], stats["benchmark_name"])
        )

    sentences.append(
        describe_tail_risk(
            stats["hist_var"],
            stats["hist_es"],
            stats["param_es"],
            stats["kurtosis"],
            stats["confidence"],
        )
    )

    window_name = WINDOW_NAMES.get(stats["window_label"], stats["window_label"])
    sentences.append(
        f"These statements describe the past {window_name} only. They do not predict "
        "future returns and are not a recommendation to buy or sell."
    )

    return sentences