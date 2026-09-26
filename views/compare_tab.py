# views/compare_tab.py
# Tab 2, "Compare stocks": the stocks in the comparison list, each rescaled to start at 100.

import streamlit as st

from charts import comparison_chart
from data import get_aligned_closes


def show_comparison() -> None:
    """Tab 2: compare the stocks in the comparison list, each rescaled to start at 100."""
    st.subheader("Compare stocks")
    st.write(
        "Each stock is rescaled to start at 100, so you can compare performance "
        "regardless of share price or currency."
    )

    watchlist = st.session_state.watchlist

    # len() counts the items in a list.
    if len(watchlist) == 0:
        st.info(
            "Your comparison list is empty. In the Explore a stock tab, search for a "
            "stock and click 'Add ... to comparison list'."
        )
        return

    # ", ".join(...) glues the list items together into one piece of text.
    st.write("**In your list:** " + ", ".join(watchlist))

    if st.button("Clear comparison list"):
        st.session_state.watchlist = []
        st.rerun()  # start a fresh run straight away so the page updates

    if len(watchlist) == 1:
        st.caption("Add at least one more stock to compare.")

    compare_range = st.segmented_control(
        "Comparison period",
        options=["1M", "1Y", "5Y"],
        default="1Y",
        key="compare_range",
    )
    if compare_range is None:
        compare_range = "1Y"

    try:
        with st.spinner("Building the comparison..."):
            closes = get_aligned_closes(watchlist, compare_range)
    except Exception:
        st.error("Couldn't download the comparison data. Check your internet connection and try again.")
        return

    if closes.empty:
        st.warning("Couldn't find overlapping price data for these stocks.")
        return

    st.plotly_chart(comparison_chart(closes, compare_range))
