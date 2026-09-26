# views/goal_planner.py
# Tab 3, "Portfolio & projection" (part 2): the goal planner, a Monte Carlo projection
# of the suggested portfolio with regular investing, fees and inflation.

import numpy as np
import pandas as pd
import streamlit as st

from charts import fan_chart
from montecarlo import (
    HORIZON_YEARS,
    LONG_RUN_SHARPE,
    bootstrap_growth_factors,
    build_paths,
    exact_percentile,
    final_value_summary,
    long_run_expected_return,
    monthly_portfolio_returns,
    normal_growth_factors,
    percentile_bands,
    total_paid_in,
)
from portfolio import daily_returns_table
from views.common import RETURN_BASES


def show_goal_planner(
    portfolio_closes: pd.DataFrame,
    suggested: np.ndarray,
    s_return: float,
    s_vol: float,
    appetite: str,
    inflation: float,
    todays_risk_free: float,
) -> None:
    """Tab 3 (continued): the goal planner, a Monte Carlo projection of the suggested
    portfolio with regular investing, fees and inflation."""
    st.divider()
    st.markdown("**Goal planner: Monte Carlo projection of the suggested portfolio**")
    st.write(
        "Simulates 10,000 possible futures for the suggested portfolio, including regular "
        "monthly investing, fees and inflation. The shaded band shows the range between the "
        "5th and 95th percentiles: in 90% of the simulations, the value ended up inside it."
    )

    # --- Settings, two per row ---
    r1_left, r1_right = st.columns(2)
    with r1_left:
        horizon = st.segmented_control(
            "Time horizon",
            options=list(HORIZON_YEARS.keys()),  # ["1Y", "5Y", "10Y", "20Y", "30Y"]
            default="10Y",
            key="mc_horizon",
        )
    with r1_right:
        method = st.segmented_control(
            "Simulation method",
            options=["Normal model", "Historical bootstrap"],
            default="Normal model",
            key="mc_method",
            help=(
                "Normal model: monthly log-returns drawn from a normal distribution with the "
                "portfolio's average return and volatility. Historical bootstrap: each simulated "
                "month re-uses a real past month of the portfolio's returns, picked at random."
            ),
        )

    # --- The return assumption: the most important input for a long-term plan ---
    long_run_return = long_run_expected_return(todays_risk_free, s_vol)
    basis = st.segmented_control(
        "Return assumption",
        options=RETURN_BASES,
        default="Long-run (cautious)",
        key="mc_return_basis",
        help=(
            "Long-run (cautious): today's risk-free rate plus a modest premium for the risk "
            f"taken ({LONG_RUN_SHARPE:.2f} x the portfolio's volatility). Historical: this "
            "portfolio's average return over the history used, which can be unrealistically "
            "high after a few strong years. Custom: your own figure."
        ),
    )
    if basis is None:
        basis = "Long-run (cautious)"

    if basis == "Custom":
        assumed_return = st.number_input(
            "Your expected return (% per year, before fees and inflation)",
            min_value=-10.0,
            max_value=30.0,
            value=round(long_run_return * 100, 1),
            step=0.5,
            key="mc_custom_return",
        ) / 100
    elif basis == "Historical":
        assumed_return = s_return
    else:
        assumed_return = long_run_return

    st.caption(
        f"Assumed return: **{assumed_return:.1%} a year** before fees and inflation "
        f"(long-run: {todays_risk_free:.2%} risk-free + {LONG_RUN_SHARPE:.2f} x "
        f"{s_vol:.1%} volatility = {long_run_return:.1%}; historical average: {s_return:.1%}). "
        f"Volatility used: {s_vol:.1%}, from the portfolio's history."
    )
    if basis == "Historical" and s_return > 0.12:
        st.warning(
            f"A historical average of {s_return:.1%} a year is unlikely to last for decades: "
            "it reflects a few unusually strong years for these particular stocks. "
            "The long-run assumption is more realistic for planning."
        )

    r2_left, r2_right = st.columns(2)
    with r2_left:
        start_value = st.number_input(
            "Starting amount (£)",
            min_value=0,
            max_value=10_000_000,
            value=10_000,
            step=1_000,
            key="mc_start_value",
        )
    with r2_right:
        monthly = st.number_input(
            "Monthly contribution (£)",
            min_value=0,
            max_value=100_000,
            value=200,
            step=50,
            key="mc_monthly",
            help="Assumed to rise with inflation, so it keeps the same buying power over time.",
        )

    r3_left, r3_right = st.columns(2)
    with r3_left:
        fee_pct = st.number_input(
            "Annual fees (%)",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.05,
            key="mc_fee",
            help="Platform and fund charges, taken from the portfolio's value every month.",
        )
    with r3_right:
        goal_value = st.number_input(
            "Your goal (£, 0 = no goal)",
            min_value=0,
            max_value=100_000_000,
            value=50_000,
            step=1_000,
            key="mc_goal",
        )

    # st.toggle draws an on/off switch and returns True or False.
    todays_money = st.toggle(
        "Show in today's money (after inflation)",
        value=True,
        key="mc_todays_money",
        help=(
            "On: amounts are shown in today's buying power, which is what matters for a goal. "
            "Off: amounts are in future pounds, which look bigger but buy less."
        ),
    )
    st.caption(
        f"Inflation assumption: {inflation:.1%} a year, set in the sidebar. "
        "Monthly contributions are assumed to rise with inflation."
    )

    if horizon is None:
        horizon = "10Y"
    if method is None:
        method = "Normal model"
    if start_value == 0 and monthly == 0:
        st.info("Enter a starting amount or a monthly contribution to run the projection.")
        return

    years = HORIZON_YEARS[horizon]
    fee = fee_pct / 100
    goal = goal_value if goal_value > 0 else None  # None means "no goal"

    # --- Run BOTH methods (the maths lives in montecarlo.py), so we can compare them ---
    with st.spinner("Simulating 10,000 possible futures..."):
        # Method 1: the normal model, using the suggested portfolio's average and volatility.
        normal_paths = build_paths(
            normal_growth_factors(assumed_return, s_vol, years),
            start_value, monthly, fee, inflation, todays_money,
        )
        # Method 2: the historical bootstrap, using the portfolio's real past monthly returns.
        past_monthly = monthly_portfolio_returns(daily_returns_table(portfolio_closes), suggested)
        # The past months are re-centred on the same assumed return, so the two
        # methods differ only in the SHAPE of the returns, not their average.
        bootstrap_result = build_paths(
            bootstrap_growth_factors(past_monthly, years, target_annual_return=assumed_return),
            start_value, monthly, fee, inflation, todays_money,
        )

    paid_in = total_paid_in(start_value, monthly, years, inflation, todays_money)

    # Show the chart and numbers for whichever method is selected.
    if method == "Normal model":
        paths = normal_paths
    else:
        paths = bootstrap_result

    bands = percentile_bands(paths, years)
    final = final_value_summary(paths, paid_in[-1], goal)

    money_note = "in today's money" if todays_money else "in future pounds"
    st.plotly_chart(
        fan_chart(
            bands,
            paths[:20],  # the first 20 rows: 20 example futures to draw faintly
            paid_in,
            goal,
            f"{method}: 10,000 futures over {years} years, {appetite.lower()} risk appetite portfolio",
            f"Portfolio value (£, {money_note})",
        )
    )

    # --- The key numbers at the end of the horizon, in a 2 x 2 grid ---
    row1_left, row1_right = st.columns(2)
    row2_left, row2_right = st.columns(2)
    row1_left.metric(
        "Bad case (5th percentile)",
        f"£{final['5th']:,.0f}",
        help="1 in 20 simulated futures ended below this value.",
    )
    row1_right.metric(
        "Median (50th percentile)",
        f"£{final['50th']:,.0f}",
        help="Half the simulated futures ended above this value, and half below.",
    )
    row2_left.metric(
        "Good case (95th percentile)",
        f"£{final['95th']:,.0f}",
        help="1 in 20 simulated futures ended above this value.",
    )
    if goal is not None:
        row2_right.metric(
            "Chance of reaching your goal",
            f"{final['chance_of_goal']:.0%}",
            help=f"The share of the 10,000 simulated futures that ended at or above £{goal:,.0f}.",
        )
    else:
        row2_right.metric(
            "Chance of ending below the total paid in",
            f"{final['chance_of_loss']:.0%}",
            help="The share of the 10,000 simulated futures that ended below the total paid in.",
        )

    st.caption(
        f"All amounts are {money_note}. Total paid in over {years} years: £{paid_in[-1]:,.0f}. "
        f"Chance of ending below the total paid in: {final['chance_of_loss']:.0%}. "
        f"After {fee_pct:.2f}% annual fees."
    )

    # --- Checks and notes ---
    lump_sum_only = monthly == 0 and fee == 0 and (not todays_money or inflation == 0)
    if method == "Normal model" and lump_sum_only:
        # Check the simulation against the exact formula for the median.
        exact_median = exact_percentile(start_value, assumed_return, s_vol, years, 50)
        st.caption(
            f"Check: for a single lump sum with no fees, the median has an exact formula, "
            f"which gives £{exact_median:,.0f}. The simulation gives £{final['50th']:,.0f}, "
            "so the two agree closely."
        )
    elif method == "Normal model":
        st.caption(
            "With regular contributions, fees or inflation, there's no simple exact formula, "
            "which is exactly why simulation is useful. (Set the contribution, fees and "
            "inflation to 0 to see the simulation checked against the exact formula.)"
        )
    else:
        st.caption(
            f"The bootstrap re-uses {len(past_monthly)} months of the portfolio's actual past "
            "returns, picked at random with replacement, shifted so their average matches the "
            "return assumption above (their real ups and downs are kept)."
        )
        if len(past_monthly) < 36:
            st.warning(
                "Fewer than 3 years of monthly history are being re-used, so the bootstrap is "
                "drawing from very few outcomes. Try the 5Y history setting in the sidebar."
            )

    # --- Compare the two methods side by side: this is "model risk" ---
    st.markdown("**How much does the choice of model matter?**")
    last_column_name = "Chance of reaching goal" if goal is not None else "Chance of a loss"
    last_column_key = "chance_of_goal" if goal is not None else "chance_of_loss"
    normal_final = final_value_summary(normal_paths, paid_in[-1], goal)
    bootstrap_final = final_value_summary(bootstrap_result, paid_in[-1], goal)

    method_table = pd.DataFrame(
        {
            "Bad case (5th)": [normal_final["5th"], bootstrap_final["5th"]],
            "Median (50th)": [normal_final["50th"], bootstrap_final["50th"]],
            "Good case (95th)": [normal_final["95th"], bootstrap_final["95th"]],
            last_column_name: [normal_final[last_column_key], bootstrap_final[last_column_key]],
        },
        index=["Normal model", "Historical bootstrap"],
    )
    # A dictionary of formats: money columns with £ and commas, the chance as a percentage.
    st.dataframe(
        method_table.style.format(
            {
                "Bad case (5th)": "£{:,.0f}",
                "Median (50th)": "£{:,.0f}",
                "Good case (95th)": "£{:,.0f}",
                last_column_name: "{:.0%}",
            }
        )
    )
    st.caption(
        "Both methods use the same past data, but make different assumptions. The gap between "
        "them is a measure of model risk: how much the answer depends on the modelling choice "
        "rather than on the data."
    )

    st.caption(
        "These projections use the return assumption above and the portfolio's past volatility, "
        "with constant weights (rebalanced) and steady fees and inflation. Returns are in "
        "each stock's own currency and are shown in £ for illustration; currency movements and "
        "taxes are not modelled. The average return itself is a very uncertain estimate. This "
        "is an educational illustration, not a forecast or a recommendation."
    )
