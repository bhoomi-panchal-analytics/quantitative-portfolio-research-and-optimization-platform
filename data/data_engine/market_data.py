# data_engine/market_data.py

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"

# yfinance-supported intervals commonly useful for this app.
VALID_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _clean_ticker(ticker: str) -> str:
    """Clean and normalize a Yahoo Finance ticker."""
    return str(ticker).strip().upper()


def _clean_tickers(tickers: Iterable[str]) -> list[str]:
    """Clean tickers and remove duplicates while preserving order."""
    cleaned = []

    for ticker in tickers:
        ticker = _clean_ticker(ticker)

        if ticker and ticker not in cleaned:
            cleaned.append(ticker)

    return cleaned


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns returned by yfinance.

    Example:
        ('Close', 'RELIANCE.NS') -> 'Close'
    """
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def _validate_interval(interval: str) -> None:
    """Validate requested Yahoo Finance interval."""
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Use one of: {sorted(VALID_INTERVALS)}"
        )


# ============================================================
# SINGLE-TICKER HISTORICAL DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_market_data(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Download historical OHLCV data for one ticker.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker, e.g. RELIANCE.NS

    period:
        Examples: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max

    interval:
        Examples: 1m, 5m, 15m, 1h, 1d, 1wk, 1mo

    Returns
    -------
    pd.DataFrame
        Columns:
        Open, High, Low, Close, Volume
    """

    ticker = _clean_ticker(ticker)
    _validate_interval(interval)

    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        data = _flatten_columns(data)

        expected_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        available_columns = [
            col for col in expected_columns
            if col in data.columns
        ]

        data = data[available_columns].copy()

        data.index = pd.to_datetime(data.index)

        data = data.sort_index()

        data = data.dropna(subset=["Close"])

        return data

    except Exception as e:
        st.warning(f"Could not load {ticker}: {e}")
        return pd.DataFrame()


# ============================================================
# MULTIPLE-TICKER DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_multiple_market_data(
    tickers: Iterable[str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for multiple tickers.

    Returns
    -------
    pd.DataFrame
        Date index with one column per ticker.
    """

    tickers = _clean_tickers(tickers)

    if not tickers:
        return pd.DataFrame()

    _validate_interval(interval)

    try:
        data = yf.download(
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        # Multiple tickers normally produce MultiIndex columns.
        if isinstance(data.columns, pd.MultiIndex):

            if "Close" in data.columns.get_level_values(0):
                data = data["Close"]

            elif "Close" in data.columns.get_level_values(-1):
                data = data.xs(
                    "Close",
                    axis=1,
                    level=-1
                )

        else:
            # Single ticker fallback
            if "Close" in data.columns:
                data = data[["Close"]]
                data.columns = [tickers[0]]

        data.index = pd.to_datetime(data.index)

        data = data.sort_index()

        data = data.dropna(how="all")

        return data

    except Exception as e:
        st.warning(f"Could not load market data: {e}")
        return pd.DataFrame()


# ============================================================
# CLOSE PRICES ONLY
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_close_prices(
    tickers: Iterable[str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Return only closing prices."""

    data = get_multiple_market_data(
        tickers=tickers,
        period=period,
        interval=interval,
    )

    if data.empty:
        return data

    return data.ffill()


# ============================================================
# RETURNS
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_returns(
    tickers: Iterable[str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Calculate percentage returns for multiple securities.
    """

    prices = get_close_prices(
        tickers=tickers,
        period=period,
        interval=interval,
    )

    if prices.empty:
        return prices

    returns = prices.pct_change()

    return returns.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(how="all")


# ============================================================
# LATEST PRICES
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_latest_prices(
    tickers: Iterable[str],
) -> pd.DataFrame:
    """
    Get the most recent available price for each ticker.

    Returns
    -------
    DataFrame:
        Ticker
        Price
        PriceDate
    """

    tickers = _clean_tickers(tickers)

    if not tickers:
        return pd.DataFrame(
            columns=["Ticker", "Price", "PriceDate"]
        )

    try:
        data = yf.download(
            tickers,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data is None or data.empty:
            return pd.DataFrame(
                columns=["Ticker", "Price", "PriceDate"]
            )

        if isinstance(data.columns, pd.MultiIndex):

            if "Close" in data.columns.get_level_values(0):
                close = data["Close"]

            else:
                close = data.xs(
                    "Close",
                    axis=1,
                    level=-1,
                )

        else:
            close = data[["Close"]]
            close.columns = [tickers[0]]

        close = close.dropna(how="all")

        if close.empty:
            return pd.DataFrame(
                columns=["Ticker", "Price", "PriceDate"]
            )

        latest = []

        for ticker in close.columns:

            series = close[ticker].dropna()

            if series.empty:
                continue

            latest.append(
                {
                    "Ticker": ticker,
                    "Price": float(series.iloc[-1]),
                    "PriceDate": series.index[-1],
                }
            )

        return pd.DataFrame(latest)

    except Exception as e:
        st.warning(f"Could not load latest prices: {e}")

        return pd.DataFrame(
            columns=["Ticker", "Price", "PriceDate"]
        )


# ============================================================
# MARKET SNAPSHOT
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_market_snapshot(
    tickers: Iterable[str],
) -> pd.DataFrame:
    """
    Return latest price, previous close and daily change.

    Useful for Dashboard cards and watchlists.
    """

    tickers = _clean_tickers(tickers)

    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):

            if "Close" in data.columns.get_level_values(0):
                close = data["Close"]

            else:
                close = data.xs(
                    "Close",
                    axis=1,
                    level=-1,
                )

        else:
            close = data[["Close"]]
            close.columns = [tickers[0]]

        close = close.dropna(how="all")

        records = []

        for ticker in close.columns:

            series = close[ticker].dropna()

            if len(series) < 1:
                continue

            latest_price = float(series.iloc[-1])

            previous_price = (
                float(series.iloc[-2])
                if len(series) >= 2
                else np.nan
            )

            change = (
                latest_price - previous_price
                if not np.isnan(previous_price)
                else np.nan
            )

            change_pct = (
                change / previous_price
                if previous_price
                and not np.isnan(previous_price)
                else np.nan
            )

            records.append(
                {
                    "Ticker": ticker,
                    "Price": latest_price,
                    "Previous Close": previous_price,
                    "Change": change,
                    "Change %": change_pct,
                    "Date": series.index[-1],
                }
            )

        return pd.DataFrame(records)

    except Exception as e:
        st.warning(f"Could not create market snapshot: {e}")
        return pd.DataFrame()


# ============================================================
# BENCHMARK DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_benchmark_data(
    benchmark: str = "^NSEI",
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Download benchmark data.

    Examples:
        ^NSEI     = NIFTY 50
        ^BSESN    = Sensex
        ^NSEBANK  = NIFTY Bank
    """

    return get_market_data(
        ticker=benchmark,
        period=period,
        interval=interval,
    )


# ============================================================
# NORMALIZED PERFORMANCE
# ============================================================

def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price series to 100.

    Useful for comparing different stocks.

    Formula:
        Normalized Price =
        Price / Starting Price × 100
    """

    if prices.empty:
        return prices

    first_valid = prices.ffill().bfill().iloc[0]

    return prices.divide(first_valid) * 100


# ============================================================
# DAILY MARKET DATA FOR PORTFOLIO
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_portfolio_market_data(
    portfolio_tickers: Iterable[str],
    benchmark: str = "^NSEI",
    period: str = DEFAULT_PERIOD,
) -> dict:
    """
    Load all market data required by the portfolio dashboard.

    Returns:
        {
            "prices": ...,
            "returns": ...,
            "benchmark": ...,
            "benchmark_returns": ...
        }
    """

    tickers = _clean_tickers(portfolio_tickers)

    prices = get_close_prices(
        tickers,
        period=period,
        interval="1d",
    )

    returns = (
        prices.pct_change()
        if not prices.empty
        else pd.DataFrame()
    )

    benchmark_data = get_benchmark_data(
        benchmark=benchmark,
        period=period,
        interval="1d",
    )

    if not benchmark_data.empty:
        benchmark_close = benchmark_data["Close"]
        benchmark_returns = benchmark_close.pct_change()
    else:
        benchmark_close = pd.Series(dtype=float)
        benchmark_returns = pd.Series(dtype=float)

    return {
        "prices": prices,
        "returns": returns,
        "benchmark": benchmark_close,
        "benchmark_returns": benchmark_returns,
    }


# ============================================================
# CLEAR CACHE
# ============================================================

def clear_market_data_cache() -> None:
    """
    Clear Streamlit's cached market data.

    Call this when the user presses a 'Refresh Data' button.
    """

    st.cache_data.clear()
