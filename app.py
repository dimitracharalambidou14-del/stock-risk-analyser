# app.py
# The entry point of the Stock Risk & Portfolio Analyser. Run it with:
#     streamlit run app.py
#
# This file only lays out the page. Each part of the page lives in its own file:
#   views/sidebar.py        - the left-hand panel: shared settings and "About"
#   views/explore_tab.py    - tab 1: search a stock, chart it, analyse its risk
#   views/compare_tab.py    - tab 2: compare stocks rescaled to start at 100
#   views/portfolio_tab.py  - tab 3: portfolio suggestion and efficient frontier
#   views/goal_planner.py   - tab 3 (continued): the Monte Carlo goal planner
#   views/strategy_tab.py   - tab 4: the out-of-sample strategy test
#   views/common.py         - settings and small tools shared by the views
#
# The maths and data live in the top-level files, with no page code in them:
#   data.py, metrics.py, portfolio.py, montecarlo.py, backtest.py, summary.py, charts.py

import streamlit as st

from views.common import DISCLAIMER, load_example_portfolio, show_data_sources_and_assumptions
from views.compare_tab import show_comparison
from views.explore_tab import show_stock_explorer
from views.portfolio_tab import show_portfolio
from views.sidebar import show_sidebar
from views.strategy_tab import show_strategy_test

# st.set_page_config sets the browser-tab title and icon. It must be the first
# Streamlit command that draws anything on the page.
st.set_page_config(page_title="Stock Risk & Portfolio Analyser", page_icon="📈")

# SESSION STATE: memory that survives reruns (for as long as this browser tab is open).
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# --- The sidebar (views/sidebar.py) returns the settings chosen there ---
window_label, risk_free, inflation = show_sidebar()

# --- Title and disclaimer ---
st.title("Stock Risk & Portfolio Analyser")
st.caption("Search for a stock, measure its risk, compare stocks, explore a portfolio and test strategies.")
st.info(DISCLAIMER, icon="ℹ️")

# One click fills the app with an example, so first-time visitors see every tab working.
st.button(
    "▶ New here? Load an example portfolio (Apple, Microsoft and Tesco)",
    on_click=load_example_portfolio,  # on_click runs the function when the button is pressed
    help="Adds three example stocks to the comparison list and analyses Apple.",
)

# --- The three tabs ---
# st.tabs returns one container per tab; "with" puts content inside it.
tab_explore, tab_compare, tab_portfolio, tab_strategy = st.tabs(
    ["Explore a stock", "Compare stocks", "Portfolio & projection", "Strategy test"]
)

with tab_explore:
    show_stock_explorer(window_label, risk_free)

with tab_compare:
    show_comparison()

with tab_portfolio:
    show_portfolio(window_label, risk_free, inflation)

with tab_strategy:
    show_strategy_test(risk_free)

# --- Which numbers are live data, and which are assumptions (views/common.py) ---
st.divider()
show_data_sources_and_assumptions()

# --- Footer ---
st.caption(
    "Educational tool, not financial advice. Figures describe the past and use free data "
    "that may be delayed or incomplete. Currency effects, fees and taxes are not modelled."
)