
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

from util import (
    normalize_symbols,
    get_prices,
    get_snapshot,
    get_fundamentals,
    technical_indicators,
    portfolio_metrics,
    optimize_portfolio,
    efficient_frontier,
    risk_contribution,
    diversification_stats,
    sentiment_from_news,
    benchmark_symbols,
    format_pct,
)

st.set_page_config(
    page_title="AlphaLab | Portfolio Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.metric-card {padding: 0.8rem 1rem; border: 1px solid rgba(128,128,128,.25);
              border-radius: 12px; background: rgba(128,128,128,.05);}
.small-note {font-size: .82rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

st.title("AlphaLab")
st.caption("Portfolio intelligence: fundamentals + technicals + sentiment + risk + optimization")

with st.sidebar:
    st.header("Portfolio")
    symbols_text = st.text_area(
        "Stocks (Yahoo/NSE symbols)",
        value="RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, BHARTIARTL.NS, LT.NS, ITC.NS",
        height=110,
    )
    symbols = normalize_symbols(symbols_text)

    lookback = st.selectbox("Historical window", ["1y", "2y", "5y", "10y"], index=1)
    benchmark = st.selectbox(
        "Benchmark",
        ["^NSEI", "^BSESN", "^NSEBANK", "^CNX100", "^CNX500"],
        index=0,
    )
    risk_free = st.number_input("Risk-free rate (%)", value=6.50, step=0.25) / 100
    refresh = st.button("Refresh market data", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Optimization controls")
    max_weight = st.slider("Maximum weight / stock", 5, 50, 20) / 100
    min_weight = st.slider("Minimum weight / stock", 0, 10, 0) / 100
    target_return = st.number_input("Target annual return (%)", value=15.0, step=1.0) / 100
    objective = st.selectbox(
        "Objective",
        ["Max Sharpe", "Minimum Volatility", "Target Return", "Risk Parity", "Maximum Diversification"],
    )

if not symbols:
    st.error("Add at least one valid ticker.")
    st.stop()

if refresh:
    st.cache_data.clear()
    st.rerun()

with st.spinner("Pulling market data and calculating analytics..."):
    prices = get_prices(symbols, lookback)
    benchmark_prices = get_prices([benchmark], lookback)
    snap = get_snapshot(symbols)

if prices.empty:
    st.error("No market data returned. Check ticker symbols or your data provider.")
    st.stop()

returns = prices.pct_change().dropna()
bench_returns = benchmark_prices[benchmark].pct_change().dropna() if benchmark in benchmark_prices else pd.Series(dtype=float)

tabs = st.tabs([
    "Dashboard",
    "Fundamental Analysis",
    "Technical Analysis",
    "Sentiment",
    "Risk Analytics",
    "Portfolio Optimization",
    "Market & Diversification",
], on_change="rerun")

# 1. Dashboard
if tabs[0].open:
    with tabs[0]:
        st.subheader("Portfolio Dashboard")

        current = prices.iloc[-1]
        prev = prices.iloc[-2] if len(prices) > 1 else current
        daily = (current / prev - 1).sort_values(ascending=False)

        c = st.columns(5)
        c[0].metric("Stocks", len(symbols))
        c[1].metric("Best today", daily.index[0], format_pct(daily.iloc[0]))
        c[2].metric("Worst today", daily.index[-1], format_pct(daily.iloc[-1]))
        c[3].metric("Portfolio volatility", format_pct(portfolio_metrics(returns, risk_free)["volatility"]))
        c[4].metric("Equal-weight Sharpe", f"{portfolio_metrics(returns, risk_free)['sharpe']:.2f}")

        st.plotly_chart(
            px.line(
                prices / prices.iloc[0] * 100,
                title="Normalized performance (100 = start)",
                labels={"value": "Indexed value", "Date": "Date"},
            ),
            use_container_width=True,
        )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.bar(daily, title="1-day return by stock", labels={"value": "Return", "index": "Ticker"}),
                use_container_width=True,
            )
        with right:
            corr = returns.corr()
            st.plotly_chart(
                px.imshow(corr, text_auto=".2f", title="Return correlation matrix", aspect="auto"),
                use_container_width=True,
            )

        st.dataframe(snap, use_container_width=True)

        st.warning(
            "This is an analytics system, not a guaranteed-return machine. "
            "Optimization can maximize a mathematical objective while losing money in the real world."
        )

# 2. Fundamentals
if tabs[1].open:
    with tabs[1]:
        st.subheader("Fundamental Analysis")

        fund = get_fundamentals(symbols)
        if fund.empty:
            st.info("Fundamental fields were not returned by the current data source.")
        else:
            st.dataframe(fund, use_container_width=True)

            numeric_candidates = ["marketCap", "trailingPE", "forwardPE", "priceToBook",
                                  "returnOnEquity", "returnOnAssets", "profitMargins",
                                  "operatingMargins", "debtToEquity", "currentRatio",
                                  "revenueGrowth", "earningsGrowth", "dividendYield"]

            available = [x for x in numeric_candidates if x in fund.columns]
            metric = st.selectbox("Compare fundamental metric", available, index=0 if available else None)

            if metric:
                chart = fund[["symbol", metric]].dropna().sort_values(metric)
                st.plotly_chart(
                    px.bar(chart, x=metric, y="symbol", orientation="h", title=f"{metric} comparison"),
                    use_container_width=True,
                )

        st.markdown("""
**Fundamental engine to add next:** revenue/EBITDA/EPS CAGR, ROIC, ROE DuPont decomposition,
FCF yield, EV/EBITDA, PEG, interest coverage, debt maturity, working-capital efficiency,
earnings quality, accruals, Piotroski F-score and Altman Z-score.
""")

# 3. Technicals
if tabs[2].open:
    with tabs[2]:
        st.subheader("Technical Analysis")

        selected = st.selectbox("Stock", symbols)
        tech = technical_indicators(prices[selected].dropna())

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=tech.index,
            open=tech["Close"],
            high=tech["Close"],
            low=tech["Close"],
            close=tech["Close"],
            name="Price",
        ))
        fig.add_trace(go.Scatter(x=tech.index, y=tech["SMA20"], name="SMA 20"))
        fig.add_trace(go.Scatter(x=tech.index, y=tech["SMA50"], name="SMA 50"))
        fig.add_trace(go.Scatter(x=tech.index, y=tech["SMA200"], name="SMA 200"))
        fig.update_layout(title=f"{selected}: price + moving averages", height=600)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.line(tech, y=["RSI14", "RSI70", "RSI30"], title="RSI(14)"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                px.line(tech, y=["MACD", "MACD_signal", "MACD_hist"], title="MACD"),
                use_container_width=True,
            )

        latest = tech.iloc[-1]
        st.dataframe(pd.DataFrame({
            "Indicator": ["RSI14", "MACD", "ATR14", "20D momentum", "50D momentum", "200D momentum"],
            "Value": [
                latest["RSI14"], latest["MACD"], latest["ATR14"],
                latest["Momentum20"], latest["Momentum50"], latest["Momentum200"]
            ],
        }), use_container_width=True)

# 4. Sentiment
if tabs[3].open:
    with tabs[3]:
        st.subheader("Sentiment & News")

        sent = sentiment_from_news(symbols)
        if sent.empty:
            st.info("No usable news sentiment was returned. Sentiment is optional and should never be treated as a price forecast.")
        else:
            st.dataframe(sent, use_container_width=True)
            st.plotly_chart(
                px.bar(sent.sort_values("sentiment"), x="sentiment", y="symbol",
                       orientation="h", title="News sentiment score"),
                use_container_width=True,
            )

            st.caption(
                "The prototype uses simple lexical sentiment. For an expert system, replace it with "
                "FinBERT/finance-specific NLP, entity linking, source weighting and event detection."
            )

# 5. Risk
if tabs[4].open:
    with tabs[4]:
        st.subheader("Risk Analytics")

        eq_weights = np.repeat(1 / len(symbols), len(symbols))
        pm = portfolio_metrics(returns, risk_free, eq_weights)
        rc = risk_contribution(returns, eq_weights)

        c = st.columns(6)
        c[0].metric("CAGR", format_pct(pm["cagr"]))
        c[1].metric("Volatility", format_pct(pm["volatility"]))
        c[2].metric("Sharpe", f"{pm['sharpe']:.2f}")
        c[3].metric("Sortino", f"{pm['sortino']:.2f}")
        c[4].metric("Max drawdown", format_pct(pm["max_drawdown"]))
        c[5].metric("VaR 95%", format_pct(pm["var95"]))

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.area(pm["drawdown"].to_frame("Drawdown"), title="Drawdown curve"),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                px.bar(rc, x=rc.index, y=rc.values, title="Risk contribution (equal weight)"),
                use_container_width=True,
            )

        st.markdown("""
**Important terminology:** Sharpe ratio is standard. “Shannon ratio” is not a standard portfolio
performance ratio. This app therefore uses Shannon entropy as a diversification statistic:
H = -Σ pᵢ ln(pᵢ). Normalized entropy is H / ln(N), where 1 means perfectly even weights.
""")

# 6. Optimization
if tabs[5].open:
    with tabs[5]:
        st.subheader("Portfolio Optimization")

        if objective == "Target Return" and target_return <= 0:
            st.warning("Target return should be positive for this setup.")

        result = optimize_portfolio(
            returns,
            risk_free=risk_free,
            objective=objective,
            max_weight=max_weight,
            min_weight=min_weight,
            target_return=target_return,
        )

        w = result["weights"]
        metrics = result["metrics"]

        c = st.columns(5)
        c[0].metric("Expected return", format_pct(metrics["return"]))
        c[1].metric("Volatility", format_pct(metrics["volatility"]))
        c[2].metric("Sharpe", f"{metrics['sharpe']:.2f}")
        c[3].metric("Max weight", format_pct(w.max()))
        c[4].metric("Entropy", f"{diversification_stats(w)['normalized_entropy']:.2f}")

        st.plotly_chart(
            px.bar(w.sort_values(ascending=False), title=f"Recommended weights: {objective}",
                   labels={"value": "Weight", "index": "Ticker"}),
            use_container_width=True,
        )

        st.dataframe(pd.DataFrame({"Ticker": w.index, "Weight": w.values}), use_container_width=True)

        frontier = efficient_frontier(returns, risk_free, max_weight, min_weight)
        if not frontier.empty:
            fig = px.scatter(
                frontier, x="volatility", y="return", color="sharpe",
                hover_data=["sharpe"], title="Approximate efficient frontier",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
**Optimization methods available in this architecture:** Mean-Variance / Max Sharpe,
minimum variance, target return, risk parity and maximum diversification. The next production
upgrade should add Black-Litterman, Hierarchical Risk Parity, CVaR optimization and transaction costs.
""")

# 7. Market
if tabs[6].open:
    with tabs[6]:
        st.subheader("Market Regime & Diversification")

        index_map = benchmark_symbols()
        idx_prices = get_prices(list(index_map.values()), lookback)
        idx_returns = idx_prices.pct_change().dropna()

        if not idx_returns.empty:
            regime = idx_returns.rolling(63).mean().iloc[-1].sort_values(ascending=False)
            st.plotly_chart(
                px.bar(regime, title="63-day average daily return by index"),
                use_container_width=True,
            )
            st.plotly_chart(
                px.line(idx_prices / idx_prices.iloc[0] * 100,
                        title="Indian market indices: normalized performance"),
                use_container_width=True,
            )

        ds = diversification_stats(eq_weights)
        st.metric("Equal-weight normalized entropy", f"{ds['normalized_entropy']:.2f}")
        st.write("1.00 = perfectly even allocation; lower values = more concentration.")

        st.markdown("""
**Market universe to extend:** Nifty 50, Nifty Next 50, Nifty 100, Nifty 200, Nifty 500,
Nifty Midcap 50/100/150, Nifty Smallcap 50/100/250, sectoral indices, Nifty Bank,
Nifty IT, Nifty Auto, Nifty Pharma, Nifty FMCG, Nifty Metal, Nifty Realty, Nifty Energy,
Nifty PSU Bank, Sensex and other BSE sector/size indices.

**Segmentation:** large/mid/small-cap classification should come from an authoritative index/
exchange constituent file rather than guessing from market cap cutoffs.
""")

st.divider()
st.caption("Data-source note: the prototype uses Yahoo Finance through yfinance. Yahoo data is not a guaranteed exchange-grade real-time feed. For true no-lag NSE/BSE data, plug in a licensed broker/exchange feed.")
