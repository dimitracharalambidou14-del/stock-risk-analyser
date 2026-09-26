# charts.py
# Everything to do with drawing charts lives in this file.
# We use Plotly, which makes interactive charts: hover to see values,
# drag to zoom in, double-click to zoom back out.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm  # the normal (bell-curve) distribution


def price_chart(prices: pd.DataFrame, ticker: str, time_range: str = "5Y") -> go.Figure:
    """Build an interactive line chart of one stock's prices.

    prices: a table of prices (daily or intraday).
    ticker: the stock symbol, used in the chart title.
    time_range: the button chosen, e.g. "1D" or "5Y".

    Returns a Plotly FIGURE (the finished chart), ready for Streamlit to display.
    """
    # 1D and 1W use intraday data (several prices per day).
    intraday = time_range in ("1D", "1W")

    if intraday:
        # Markets close overnight and at weekends. On a normal time axis,
        # those closed hours would appear as long flat lines. Instead we turn
        # each time into a text label (e.g. "Fri 25 Sep 14:30") and use a
        # "category" axis, which places each price right after the previous one,
        # so closed periods take up no space.
        x_values = prices.index.strftime("%a %d %b %H:%M")
        x_title = "Date and time (exchange's local time)"
    else:
        x_values = prices.index
        x_title = "Date"

    # A Figure is the whole chart: the canvas plus everything drawn on it.
    fig = go.Figure()

    # A "trace" is one set of data drawn on the chart - here, one line.
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=prices["Close"],
            mode="lines",
            name=ticker,
        )
    )

    # The "layout" controls everything that isn't data: titles, axes, size.
    fig.update_layout(
        title=f"{ticker}: {time_range} price (adjusted for dividends and splits)",
        xaxis_title=x_title,
        yaxis_title="Price (in the stock's own currency)",
        hovermode="x unified",  # hovering shows the value at that point
        height=450,             # chart height in pixels
    )

    if intraday:
        # Treat the x-axis labels as categories, and show at most 8 of them
        # so they don't overlap.
        fig.update_xaxes(type="category", nticks=8)

    return fig


def rebase_to_100(closes: pd.DataFrame) -> pd.DataFrame:
    """Rescale every column so it starts at exactly 100.

    Formula for each stock:  rebased value = price today / first price x 100

    So 120 means "up 20% since the start" and 90 means "down 10%",
    regardless of the share price level or currency.
    """
    # closes.iloc[0] is the first row (each stock's starting price).
    # pandas divides EVERY column by its own starting price in one go - no loop needed.
    return closes / closes.iloc[0] * 100


def comparison_chart(closes: pd.DataFrame, time_range: str) -> go.Figure:
    """Build a chart comparing several stocks, each rescaled to start at 100.

    closes: a table of closing prices, one column per ticker (lined up by date).
    time_range: "1M", "1Y" or "5Y", used in the title.
    """
    rebased = rebase_to_100(closes)

    fig = go.Figure()

    # Add one line (trace) for each ticker. "rebased.columns" lists the tickers.
    for ticker in rebased.columns:
        fig.add_trace(
            go.Scatter(
                x=rebased.index,
                y=rebased[ticker],
                mode="lines",
                name=ticker,
            )
        )

    # A dotted horizontal line at 100 marks the starting value.
    fig.add_hline(y=100, line_dash="dot", line_color="grey")

    fig.update_layout(
        title=f"Growth of 100 over {time_range} (total return, each in its own currency)",
        xaxis_title="Date",
        yaxis_title="Value (start = 100)",
        hovermode="x unified",
        legend_title_text="Ticker",
        height=450,
    )

    return fig


def drawdown_chart(drawdowns: pd.Series, ticker: str) -> go.Figure:
    """An "underwater" chart: how far below its previous peak the stock was each day.

    drawdowns: the output of drawdown_series (0 at new highs, negative otherwise).
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns * 100,      # convert decimals to percentages
            mode="lines",
            fill="tozeroy",         # shade the area between the line and zero
            line_color="firebrick",
            name="Drawdown",
        )
    )

    fig.update_layout(
        title=f"{ticker}: fall from previous peak (drawdown)",
        xaxis_title="Date",
        yaxis_title="Below previous peak",
        hovermode="x unified",
        height=350,
    )
    # Put a % sign after every number on the vertical axis.
    fig.update_yaxes(ticksuffix="%")

    return fig


def beta_chart(
    returns: pd.DataFrame, beta_value: float, ticker: str, benchmark_name: str
) -> go.Figure:
    """Scatter chart of daily returns: stock (up the side) vs market (along the bottom).

    Each dot is one day. The line of best fit has slope = beta.
    returns: the output of aligned_returns (columns "stock" and "market").
    """
    fig = go.Figure()

    # One dot per day.
    fig.add_trace(
        go.Scatter(
            x=returns["market"] * 100,
            y=returns["stock"] * 100,
            mode="markers",
            marker=dict(size=5, opacity=0.5),
            name="Daily returns",
        )
    )

    # The line of best fit: stock return = intercept + beta x market return.
    # A best-fit line always passes through the averages of x and y,
    # which gives us the intercept.
    intercept = returns["stock"].mean() - beta_value * returns["market"].mean()
    x_low = returns["market"].min()
    x_high = returns["market"].max()

    # A straight line only needs two points: one at each end.
    fig.add_trace(
        go.Scatter(
            x=[x_low * 100, x_high * 100],
            y=[(intercept + beta_value * x_low) * 100, (intercept + beta_value * x_high) * 100],
            mode="lines",
            line_color="firebrick",
            name=f"Best-fit line (slope = beta = {beta_value:.2f})",
        )
    )

    fig.update_layout(
        title=f"{ticker} vs {benchmark_name}: daily returns",
        xaxis_title=f"{benchmark_name} daily return",
        yaxis_title=f"{ticker} daily return",
        height=450,
    )
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%")

    return fig


def var_chart(
    returns: pd.Series, hist_var: float, param_var: float, ticker: str, confidence_label: str
) -> go.Figure:
    """Histogram of actual daily returns, with a normal curve and both VaR levels marked.

    It shows where the real data has "fatter tails" than the normal distribution.
    """
    fig = go.Figure()

    # A histogram groups the daily returns into bars (bins) and counts them.
    # histnorm="probability density" scales the bars so they're comparable
    # with the height of the normal curve.
    fig.add_trace(
        go.Histogram(
            x=returns,
            histnorm="probability density",
            nbinsx=80,
            opacity=0.6,
            name="Actual daily returns",
        )
    )

    # The normal curve with the SAME average and volatility as the actual returns.
    # np.linspace makes 200 evenly spaced x values from the worst to the best day.
    x_values = np.linspace(returns.min(), returns.max(), 200)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=norm.pdf(x_values, returns.mean(), returns.std()),
            mode="lines",
            line_color="darkorange",
            name="Normal distribution (same average and volatility)",
        )
    )

    # Vertical lines where each VaR sits (VaR is a loss, so it's at MINUS VaR).
    # The two lines are often very close together, so put each label on the
    # OUTER side of its own line: the line further left gets its label on the left.
    if param_var > hist_var:
        # The parametric line is further left.
        param_label_side, hist_label_side = "top left", "top right"
    else:
        # The historical line is further left (or they're equal).
        param_label_side, hist_label_side = "top right", "top left"

    fig.add_vline(
        x=-hist_var,
        line_dash="dash",
        line_color="firebrick",
        annotation_text="Historical VaR",
        annotation_position=hist_label_side,
    )
    fig.add_vline(
        x=-param_var,
        line_dash="dot",
        line_color="darkorange",
        annotation_text="Parametric VaR",
        annotation_position=param_label_side,
    )

    fig.update_layout(
        title=f"{ticker}: distribution of daily returns ({confidence_label} VaR marked)",
        xaxis_title="Daily return",
        yaxis_title="Density",
        bargap=0.05,
        height=450,
    )
    # Show the x-axis numbers as percentages, e.g. -5%.
    fig.update_xaxes(tickformat=".0%")

    return fig


def correlation_heatmap(correlations: pd.DataFrame) -> go.Figure:
    """A colour-coded grid showing the correlation between each pair of stocks.

    Red = move closely together (less diversification benefit);
    blue = move independently or in opposite directions (more benefit).
    """
    fig = go.Figure(
        go.Heatmap(
            z=correlations.values,
            x=list(correlations.columns),
            y=list(correlations.index),
            zmin=-1,                  # fix the colour scale from -1 ...
            zmax=1,                   # ... to +1
            colorscale="RdBu",
            reversescale=True,        # flip it so high correlation is red
            texttemplate="%{z:.2f}",  # write the number inside each square
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="How the stocks move together: correlation of daily returns",
        height=400,
    )
    # Plotly draws the first row at the BOTTOM by default. Reversing the y-axis puts
    # it at the top, so the grid reads like a matrix (diagonal top-left to bottom-right).
    fig.update_yaxes(autorange="reversed")

    return fig


def weights_chart(weights: pd.Series, title: str) -> go.Figure:
    """Horizontal bar chart showing how a portfolio is split between stocks.

    weights: one weight per stock (decimals adding up to 1), labelled by ticker.
    """
    fig = go.Figure(
        go.Bar(
            x=weights.values * 100,   # convert decimals to percentages
            y=list(weights.index),
            orientation="h",          # "h" = horizontal bars
            texttemplate="%{x:.1f}%", # write the percentage on each bar
            textposition="auto",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Share of the portfolio",
        # Make the chart taller when there are more stocks.
        height=150 + 45 * len(weights),
    )
    fig.update_xaxes(ticksuffix="%", range=[0, 100])

    return fig


def efficient_frontier_chart(
    frontier: pd.DataFrame,
    cloud: pd.DataFrame,
    stock_points: pd.DataFrame,
    highlights: dict[str, tuple[float, float]],
    risk_free_rate: float,
) -> go.Figure:
    """The efficient frontier, with random portfolios, single stocks and key portfolios.

    frontier: table with "Volatility" and "Expected return" (one row per point).
    cloud: random portfolios, with "Volatility", "Expected return" and "Sharpe ratio".
    stock_points: one row per stock (index = ticker), same two columns.
    highlights: {name: (volatility, expected return)} for portfolios to mark with stars.
                It must include "Maximum Sharpe ratio".
    risk_free_rate: used to draw the capital market line.
    """
    fig = go.Figure()

    # 1. The cloud of random portfolios, coloured by Sharpe ratio.
    fig.add_trace(
        go.Scatter(
            x=cloud["Volatility"] * 100,
            y=cloud["Expected return"] * 100,
            mode="markers",
            marker=dict(
                size=4,
                opacity=0.5,
                color=cloud["Sharpe ratio"],  # colour each dot by its Sharpe ratio
                colorscale="Viridis",
                showscale=True,               # show the colour key on the right
                colorbar=dict(title="Sharpe"),
            ),
            name="Random portfolios (no cap)",
            hoverinfo="skip",                 # no hover boxes for 2,000 dots
        )
    )

    # 2. The efficient frontier itself.
    fig.add_trace(
        go.Scatter(
            x=frontier["Volatility"] * 100,
            y=frontier["Expected return"] * 100,
            mode="lines",
            line=dict(width=3, color="firebrick"),
            name="Efficient frontier",
        )
    )

    # 3. The capital market line: mixes of cash (at the risk-free rate) and the
    #    maximum-Sharpe portfolio. It's a straight line starting at the
    #    risk-free rate, touching the frontier at the maximum-Sharpe portfolio.
    tangency_vol, tangency_return = highlights["Maximum Sharpe ratio"]
    slope = (tangency_return - risk_free_rate) / tangency_vol  # = the max Sharpe ratio
    end_vol = tangency_vol * 1.3  # draw the line a little past the touching point
    fig.add_trace(
        go.Scatter(
            x=[0, end_vol * 100],
            y=[risk_free_rate * 100, (risk_free_rate + slope * end_vol) * 100],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Capital market line (cash + max-Sharpe mix)",
        )
    )

    # 4. Each individual stock, as a labelled diamond.
    fig.add_trace(
        go.Scatter(
            x=stock_points["Volatility"] * 100,
            y=stock_points["Expected return"] * 100,
            mode="markers+text",
            text=list(stock_points.index),
            textposition="top center",
            marker=dict(size=11, symbol="diamond"),
            name="Individual stocks",
        )
    )

    # 5. The highlighted portfolios, as stars.
    # Several portfolios can sit in almost the same spot, so the stars get
    # different sizes (biggest first) and a white outline: when they overlap,
    # you see stars stacked inside each other instead of one hiding the others.
    star_sizes = [26, 21, 16, 11]

    # zip() pairs each (name, point) with a size.
    # "(name, (vol, ret))" unpacks the name and the (volatility, return) pair.
    for (name, (vol, ret)), size in zip(highlights.items(), star_sizes):
        fig.add_trace(
            go.Scatter(
                x=[vol * 100],
                y=[ret * 100],
                mode="markers",
                marker=dict(size=size, symbol="star", line=dict(width=1, color="white")),
                name=name,
                # What appears when you hover over the star.
                hovertemplate=(
                    name
                    + "<br>Volatility: %{x:.1f}%<br>Expected return: %{y:.1f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Efficient frontier: the best expected return for each level of risk",
        xaxis_title="Volatility (annual risk)",
        yaxis_title="Expected return (annual)",
        height=600,
        # Put the legend underneath the chart, laid out horizontally.
        legend=dict(orientation="h", yanchor="top", y=-0.2),
    )
    fig.update_xaxes(ticksuffix="%", rangemode="tozero")  # start the x-axis at 0
    fig.update_yaxes(ticksuffix="%")

    return fig


def fan_chart(
    bands: pd.DataFrame,
    sample_paths: np.ndarray,
    paid_in: np.ndarray,
    goal: float | None,
    title: str,
    y_title: str = "Portfolio value (£)",
) -> go.Figure:
    """A "fan chart" of simulated future values.

    bands: the output of percentile_bands (columns "5th", "50th", "95th"; index = years).
    sample_paths: a few individual simulated paths, drawn faintly for illustration.
    paid_in: the running total paid in, month by month (drawn as a dotted line).
    goal: the target amount (drawn as a dashed line), or None for no goal.
    """
    fig = go.Figure()
    time_in_years = bands.index

    # 1. A few individual simulated futures, drawn faintly in grey.
    #    Each one is just ONE possible future; the bands summarise all 10,000.
    for path in sample_paths:
        fig.add_trace(
            go.Scatter(
                x=time_in_years,
                y=path,
                mode="lines",
                line=dict(width=1, color="lightgrey"),
                showlegend=False,   # don't list 20 grey lines in the legend
                hoverinfo="skip",
            )
        )

    # 2. The shaded 5th-95th percentile band. We draw the 95th percentile line,
    #    then the 5th percentile line with fill="tonexty", which shades the area
    #    between it and the line drawn just before it (the 95th).
    fig.add_trace(
        go.Scatter(
            x=time_in_years,
            y=bands["95th"],
            mode="lines",
            line=dict(width=1, color="royalblue"),
            name="95th percentile (good case)",
            legendrank=1,  # legendrank sets the order in the legend: 1 = first
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_in_years,
            y=bands["5th"],
            mode="lines",
            line=dict(width=1, color="royalblue"),
            fill="tonexty",
            fillcolor="rgba(65, 105, 225, 0.2)",  # royal blue, 20% opaque
            name="5th percentile (bad case)",
            legendrank=3,
        )
    )

    # 3. The median (50th percentile) path, drawn thicker.
    fig.add_trace(
        go.Scatter(
            x=time_in_years,
            y=bands["50th"],
            mode="lines",
            line=dict(width=3, color="royalblue"),
            name="50th percentile (median)",
            legendrank=2,
        )
    )

    # 4. The total paid in so far: a dotted line that rises with each contribution.
    fig.add_trace(
        go.Scatter(
            x=time_in_years,
            y=paid_in,
            mode="lines",
            line=dict(width=2, color="grey", dash="dot"),
            name="Total paid in",
            legendrank=4,
        )
    )

    # 5. The goal, if there is one, as a dashed horizontal line.
    if goal is not None:
        fig.add_hline(
            y=goal,
            line_dash="dash",
            line_color="seagreen",
            annotation_text="Your goal",
            annotation_position="top left",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Years from now",
        yaxis_title=y_title,
        hovermode="x unified",
        height=500,
        # Plotly automatically REVERSES the legend when a chart has a shaded area.
        # "normal" switches that off, so the legendrank order above is used:
        # 95th at the top, then the median, then the 5th, matching the chart.
        legend=dict(traceorder="normal"),
    )
    # Show the values with thousands separators, e.g. 12,500.
    fig.update_yaxes(tickformat=",.0f")

    return fig


def backtest_chart(wealth_table: pd.DataFrame) -> go.Figure:
    """Growth of 100 for each strategy in the walk-forward (out-of-sample) test.

    wealth_table: one column per strategy, daily values starting at 100.
    Equal weight is drawn as a thick dashed line, because it's the benchmark
    every other strategy has to beat.
    """
    fig = go.Figure()

    for name in wealth_table.columns:
        is_benchmark = name == "Equal weight"  # True for the benchmark, False otherwise
        fig.add_trace(
            go.Scatter(
                x=wealth_table.index,
                y=wealth_table[name],
                mode="lines",
                name=name,
                # "X if condition else Y" picks X when the condition is true.
                line=dict(width=3 if is_benchmark else 1.5, dash="dash" if is_benchmark else "solid"),
            )
        )

    # A dotted line at 100 marks the starting value.
    fig.add_hline(y=100, line_dash="dot", line_color="grey")

    fig.update_layout(
        title="Out-of-sample test: growth of 100 for each strategy (after trading costs)",
        xaxis_title="Date",
        yaxis_title="Value (start = 100)",
        hovermode="x unified",
        height=500,
        # Put the long strategy names underneath the chart.
        legend=dict(orientation="h", yanchor="top", y=-0.2),
    )

    return fig