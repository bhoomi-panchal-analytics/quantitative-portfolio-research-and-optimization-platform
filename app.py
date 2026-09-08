import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Portfolio Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.title("Portfolio Intelligence Platform")
st.caption("Indian Equity Research & Portfolio Analytics")

st.divider()

# Stock selection
stock = st.selectbox(
    "Select Stock",
    [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "ITC.NS",
        "LT.NS",
        "SBIN.NS"
    ]
)

period = st.selectbox(
    "Historical Period",
    [
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "3y",
        "5y",
        "10y"
    ],
    index=3
)

if st.button("Load Market Data"):

    data = yf.download(
        stock,
        period=period,
        auto_adjust=False
    )

    if data.empty:
        st.error("No market data was returned.")
    else:
        st.success(f"Market data loaded for {stock}")

        st.subheader("Price Chart")

        st.line_chart(
            data["Close"]
        )

        st.subheader("Latest Market Data")

        st.dataframe(
            data.tail(10),
            use_container_width=True
        )
