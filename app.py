import streamlit as st

st.set_page_config(
    page_title="Portfolio Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.title("Portfolio Intelligence Platform")

st.write(
    "Quantitative analysis and portfolio optimization "
    "for Indian equities."
)

st.divider()

st.subheader("Project Status")

st.success("Level 1: Streamlit application successfully initialized.")

st.write(
    "This platform will eventually include market data, "
    "stock analysis, risk modelling, portfolio optimization, "
    "Monte Carlo simulation, and backtesting."
)
