# data_engine/fundamentals.py

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _clean_ticker(ticker: str) -> str:
    """Clean Yahoo Finance ticker."""
    return str(ticker).strip().upper()


def _safe_float(value):
    """Convert a value to float safely."""
    try:
        if value is None:
            return np.nan

        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "")

        return float(value)

    except (ValueError, TypeError):
        return np.nan


def _safe_get(info: dict, key: str):
    """Safely retrieve a Yahoo Finance information field."""
    return info.get(key, np.nan)


# ============================================================
# SINGLE COMPANY FUNDAMENTALS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(ticker: str) -> dict:
    """
    Retrieve fundamental information for one company.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker, e.g. HDFCBANK.NS

    Returns
    -------
    dict
        Fundamental metrics for the company.
    """

    ticker = _clean_ticker(ticker)

    try:
        stock = yf.Ticker(ticker)

        info = stock.info

        if not info:
            return {}

        fundamentals = {
            "Ticker": ticker,

            # ------------------------------------------------
            # Company Information
            # ------------------------------------------------
            "Company": _safe_get(info, "longName"),
            "Sector": _safe_get(info, "sector"),
            "Industry": _safe_get(info, "industry"),
            "Country": _safe_get(info, "country"),

            # ------------------------------------------------
            # Market Data
            # ------------------------------------------------
            "Market Cap": _safe_float(
                _safe_get(info, "marketCap")
            ),

            "Enterprise Value": _safe_float(
                _safe_get(info, "enterpriseValue")
            ),

            "Beta": _safe_float(
                _safe_get(info, "beta")
            ),

            # ------------------------------------------------
            # Valuation
            # ------------------------------------------------
            "PE": _safe_float(
                _safe_get(info, "trailingPE")
            ),

            "Forward PE": _safe_float(
                _safe_get(info, "forwardPE")
            ),

            "PEG": _safe_float(
                _safe_get(info, "pegRatio")
            ),

            "Price to Book": _safe_float(
                _safe_get(info, "priceToBook")
            ),

            "Price to Sales": _safe_float(
                _safe_get(info, "priceToSalesTrailing12Months")
            ),

            "EV to EBITDA": _safe_float(
                _safe_get(info, "enterpriseToEbitda")
            ),

            "EV to Revenue": _safe_float(
                _safe_get(info, "enterpriseToRevenue")
            ),

            # ------------------------------------------------
            # Profitability
            # ------------------------------------------------
            "ROE": _safe_float(
                _safe_get(info, "returnOnEquity")
            ),

            "ROA": _safe_float(
                _safe_get(info, "returnOnAssets")
            ),

            "Profit Margin": _safe_float(
                _safe_get(info, "profitMargins")
            ),

            "Operating Margin": _safe_float(
                _safe_get(info, "operatingMargins")
            ),

            "Gross Margin": _safe_float(
                _safe_get(info, "grossMargins")
            ),

            # ------------------------------------------------
            # Growth
            # ------------------------------------------------
            "Revenue Growth": _safe_float(
                _safe_get(info, "revenueGrowth")
            ),

            "Earnings Growth": _safe_float(
                _safe_get(info, "earningsGrowth")
            ),

            "Earnings Quarterly Growth": _safe_float(
                _safe_get(info, "earningsQuarterlyGrowth")
            ),

            # ------------------------------------------------
            # Balance Sheet
            # ------------------------------------------------
            "Debt to Equity": _safe_float(
                _safe_get(info, "debtToEquity")
            ),

            "Current Ratio": _safe_float(
                _safe_get(info, "currentRatio")
            ),

            "Quick Ratio": _safe_float(
                _safe_get(info, "quickRatio")
            ),

            # ------------------------------------------------
            # Cash Flow
            # ------------------------------------------------
            "Free Cash Flow": _safe_float(
                _safe_get(info, "freeCashflow")
            ),

            "Operating Cash Flow": _safe_float(
                _safe_get(info, "operatingCashflow")
            ),

            # ------------------------------------------------
            # Dividends
            # ------------------------------------------------
            "Dividend Yield": _safe_float(
                _safe_get(info, "dividendYield")
            ),

            "Dividend Rate": _safe_float(
                _safe_get(info, "dividendRate")
            ),

            "Payout Ratio": _safe_float(
                _safe_get(info, "payoutRatio")
            ),

            # ------------------------------------------------
            # Trading Range
            # ------------------------------------------------
            "52W High": _safe_float(
                _safe_get(info, "fiftyTwoWeekHigh")
            ),

            "52W Low": _safe_float(
                _safe_get(info, "fiftyTwoWeekLow")
            ),

            "50D Average": _safe_float(
                _safe_get(info, "fiftyDayAverage")
            ),

            "200D Average": _safe_float(
                _safe_get(info, "twoHundredDayAverage")
            ),

            # ------------------------------------------------
            # Shares
            # ------------------------------------------------
            "Shares Outstanding": _safe_float(
                _safe_get(info, "sharesOutstanding")
            ),

            "Float Shares": _safe_float(
                _safe_get(info, "floatShares")
            ),
        }

        return fundamentals

    except Exception as e:

        st.warning(
            f"Could not load fundamentals for {ticker}: {e}"
        )

        return {}


# ============================================================
# MULTIPLE COMPANIES
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_multiple_fundamentals(
    tickers: Iterable[str],
) -> pd.DataFrame:
    """
    Retrieve fundamentals for multiple companies.

    Returns
    -------
    DataFrame
        One row per company.
    """

    tickers = [
        _clean_ticker(ticker)
        for ticker in tickers
        if str(ticker).strip()
    ]

    # Remove duplicates
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        return pd.DataFrame()

    records = []

    for ticker in tickers:

        data = get_fundamentals(ticker)

        if data:
            records.append(data)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# ============================================================
# MARKET CAP CLASSIFICATION
# ============================================================

def classify_market_cap(
    market_cap: float,
) -> str:
    """
    Rough market-cap classification.

    IMPORTANT:
    This is NOT the official SEBI classification.

    It is intended only for dashboard grouping.
    """

    if pd.isna(market_cap):
        return "Unknown"

    # Market cap is received in INR.
    # Convert to crore.
    market_cap_crore = market_cap / 10_000_000

    if market_cap_crore >= 50_000:
        return "Large Cap"

    elif market_cap_crore >= 20_000:
        return "Mid Cap"

    else:
        return "Small Cap"


# ============================================================
# ADD MARKET CAP CATEGORY
# ============================================================

def add_market_cap_category(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a rough market-cap category to fundamentals.
    """

    if fundamentals.empty:
        return fundamentals

    fundamentals = fundamentals.copy()

    fundamentals["Market Cap Category"] = (
        fundamentals["Market Cap"]
        .apply(classify_market_cap)
    )

    return fundamentals


# ============================================================
# FUNDAMENTAL FACTOR SCORES
# ============================================================

def _percentile_score(
    series: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """
    Convert a metric into a percentile score from 0-100.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    if numeric.notna().sum() <= 1:
        return pd.Series(
            50.0,
            index=series.index
        )

    if higher_is_better:

        score = numeric.rank(
            pct=True,
            method="average"
        ) * 100

    else:

        score = (
            1
            - numeric.rank(
                pct=True,
                method="average"
            )
            + 1 / len(numeric)
        ) * 100

    return score.clip(0, 100)


def compute_factor_scores(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Value, Quality, Growth and Safety scores.

    Scores range from 0 to 100.

    Higher score = better relative position
    within the selected universe.
    """

    if fundamentals.empty:
        return fundamentals

    df = fundamentals.copy()

    # ========================================================
    # VALUE
    # Lower valuation multiples are generally better.
    # ========================================================

    value_scores = pd.concat(
        [
            _percentile_score(
                df["PE"],
                higher_is_better=False,
            ),

            _percentile_score(
                df["Forward PE"],
                higher_is_better=False,
            ),

            _percentile_score(
                df["Price to Book"],
                higher_is_better=False,
            ),

            _percentile_score(
                df["EV to EBITDA"],
                higher_is_better=False,
            ),
        ],
        axis=1,
    )

    df["Value Score"] = value_scores.mean(
        axis=1,
        skipna=True,
    )

    # ========================================================
    # QUALITY
    # Higher profitability and stronger balance sheet.
    # ========================================================

    quality_scores = pd.concat(
        [
            _percentile_score(
                df["ROE"],
                higher_is_better=True,
            ),

            _percentile_score(
                df["ROA"],
                higher_is_better=True,
            ),

            _percentile_score(
                df["Profit Margin"],
                higher_is_better=True,
            ),

            _percentile_score(
                df["Operating Margin"],
                higher_is_better=True,
            ),

            _percentile_score(
                df["Current Ratio"],
                higher_is_better=True,
            ),

            _percentile_score(
                df["Debt to Equity"],
                higher_is_better=False,
            ),
        ],
        axis=1,
    )

    df["Quality Score"] = quality_scores.mean(
        axis=1,
        skipna=True,
    )

    # ========================================================
    # GROWTH
    # ========================================================

    growth_scores = pd.concat(
        [
            _percentile_score(
                df["Revenue Growth"],
                higher_is_better=True,
            ),

            _percentile_score(
                df["Earnings Growth"],
                higher_is_better=True,
            ),

            _percentile_score(
                df["Earnings Quarterly Growth"],
                higher_is_better=True,
            ),
        ],
        axis=1,
    )

    df["Growth Score"] = growth_scores.mean(
        axis=1,
        skipna=True,
    )

    # ========================================================
    # SAFETY
    # ========================================================

    safety_scores = pd.concat(
        [
            _percentile_score(
                df["Debt to Equity"],
                higher_is_better=False,
            ),

            _percentile_score(
                df["Current Ratio"],
                higher_is_better=True,
            ),

            _percentile_score(
                df["Beta"],
                higher_is_better=False,
            ),
        ],
        axis=1,
    )

    df["Safety Score"] = safety_scores.mean(
        axis=1,
        skipna=True,
    )

    # ========================================================
    # COMPOSITE SCORE
    # ========================================================

    df["Composite Score"] = (
        df["Value Score"] * 0.25
        + df["Quality Score"] * 0.30
        + df["Growth Score"] * 0.25
        + df["Safety Score"] * 0.20
    )

    return df


# ============================================================
# FUNDAMENTAL RANKING
# ============================================================

def rank_stocks(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank stocks by composite fundamental score.
    """

    if fundamentals.empty:
        return fundamentals

    df = fundamentals.copy()

    if "Composite Score" not in df.columns:
        df = compute_factor_scores(df)

    df["Fundamental Rank"] = (
        df["Composite Score"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype("Int64")
    )

    return df.sort_values(
        "Composite Score",
        ascending=False
    )


# ============================================================
# FORMATTING
# ============================================================

def format_fundamentals(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a cleaner DataFrame for Streamlit display.
    """

    if fundamentals.empty:
        return fundamentals

    df = fundamentals.copy()

    percentage_columns = [
        "ROE",
        "ROA",
        "Profit Margin",
        "Operating Margin",
        "Gross Margin",
        "Revenue Growth",
        "Earnings Growth",
        "Earnings Quarterly Growth",
        "Dividend Yield",
        "Payout Ratio",
    ]

    for column in percentage_columns:

        if column in df.columns:
            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce"
                ) * 100
            )

    return df


# ============================================================
# CLEAR FUNDAMENTAL CACHE
# ============================================================

def clear_fundamental_cache() -> None:
    """Clear cached fundamental data."""
    st.cache_data.clear()
