
import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


def normalize_symbols(text):
    if isinstance(text, str):
        raw = [x.strip().upper() for x in text.replace("\n", ",").split(",")]
    else:
        raw = list(text)
    return list(dict.fromkeys([x for x in raw if x]))


@__import__("functools").lru_cache(maxsize=32)
def _download_cached(symbols_tuple, period):
    symbols = list(symbols_tuple)
    return yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )


def get_prices(symbols, period="2y"):
    symbols = normalize_symbols(symbols)
    if not symbols:
        return pd.DataFrame()

    raw = _download_cached(tuple(symbols), period)

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance usually returns Price x Ticker
        if "Close" in raw.columns.get_level_values(0):
            px = raw["Close"].copy()
        elif "Close" in raw.columns.get_level_values(1):
            px = raw.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        px = raw[["Close"]].copy()
        px.columns = [symbols[0]]

    px = px.dropna(how="all")
    return px.ffill()


@__import__("functools").lru_cache(maxsize=64)
def _info_cached(symbol):
    try:
        return yf.Ticker(symbol).info
    except Exception:
        return {}


def get_snapshot(symbols):
    rows = []
    for s in symbols:
        info = _info_cached(s)
        rows.append({
            "symbol": s,
            "price": info.get("currentPrice", info.get("regularMarketPrice", np.nan)),
            "market_cap": info.get("marketCap", np.nan),
            "pe": info.get("trailingPE", np.nan),
            "forward_pe": info.get("forwardPE", np.nan),
            "52w_high": info.get("fiftyTwoWeekHigh", np.nan),
            "52w_low": info.get("fiftyTwoWeekLow", np.nan),
            "beta": info.get("beta", np.nan),
        })
    return pd.DataFrame(rows)


def get_fundamentals(symbols):
    rows = []
    fields = [
        "marketCap", "enterpriseValue", "trailingPE", "forwardPE", "priceToBook",
        "enterpriseToEbitda", "returnOnEquity", "returnOnAssets",
        "profitMargins", "operatingMargins", "grossMargins", "debtToEquity",
        "currentRatio", "quickRatio", "revenueGrowth", "earningsGrowth",
        "dividendYield", "freeCashflow", "totalCash", "totalDebt",
    ]
    for s in symbols:
        info = _info_cached(s)
        row = {"symbol": s}
        for f in fields:
            row[f] = info.get(f, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def technical_indicators(close):
    df = pd.DataFrame({"Close": close.astype(float)})
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    tr = pd.concat([
        df["Close"].diff().abs(),
        (df["Close"] - df["Close"].shift(1)).abs(),
        (df["Close"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    for n in [20, 50, 200]:
        df[f"Momentum{n}"] = df["Close"].pct_change(n)

    return df


def portfolio_metrics(returns, risk_free=0.0, weights=None):
    r = returns.dropna(how="all").copy()
    r = r.dropna(axis=1, how="all")

    if weights is None:
        weights = np.repeat(1 / r.shape[1], r.shape[1])
    weights = np.asarray(weights)

    p = r.dot(weights)
    ann_return = (1 + p).prod() ** (TRADING_DAYS / max(len(p), 1)) - 1
    ann_vol = p.std(ddof=1) * np.sqrt(TRADING_DAYS)

    excess = ann_return - risk_free
    sharpe = excess / ann_vol if ann_vol > 0 else np.nan

    downside = p[p < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    sortino = excess / downside if downside > 0 else np.nan

    wealth = (1 + p).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    max_dd = drawdown.min()

    var95 = np.quantile(p, 0.05)
    cvar95 = p[p <= var95].mean() if (p <= var95).any() else var95

    return {
        "return": ann_return,
        "cagr": ann_return,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "var95": var95,
        "cvar95": cvar95,
        "drawdown": drawdown,
    }


def _annual_stats(returns):
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    return mu, cov


def _portfolio_stats(w, mu, cov, rf):
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0)))
    sharpe = (ret - rf) / vol if vol > 1e-12 else -1e9
    return ret, vol, sharpe


def optimize_portfolio(
    returns,
    risk_free=0.0,
    objective="Max Sharpe",
    max_weight=0.20,
    min_weight=0.0,
    target_return=0.15,
):
    r = returns.dropna()
    cols = r.columns.tolist()
    n = len(cols)
    mu, cov = _annual_stats(r)

    x0 = np.repeat(1 / n, n)
    bounds = [(min_weight, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if objective == "Target Return":
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - target_return})

    if objective == "Max Sharpe":
        fun = lambda w: -_portfolio_stats(w, mu, cov, risk_free)[2]
    elif objective == "Minimum Volatility":
        fun = lambda w: _portfolio_stats(w, mu, cov, risk_free)[1]
    elif objective == "Target Return":
        fun = lambda w: _portfolio_stats(w, mu, cov, risk_free)[1]
    elif objective == "Risk Parity":
        def fun(w):
            port_var = max(w @ cov @ w, 1e-12)
            mrc = cov @ w / np.sqrt(port_var)
            rc = w * mrc
            return np.sum((rc - rc.mean()) ** 2)
    else:  # Maximum Diversification
        vols = np.sqrt(np.diag(cov))
        fun = lambda w: -(w @ vols) / np.sqrt(max(w @ cov @ w, 1e-12))

    res = minimize(
        fun, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if not res.success:
        w = x0
    else:
        w = res.x

    w = pd.Series(w, index=cols)
    ret, vol, sharpe = _portfolio_stats(w.values, mu, cov, risk_free)

    return {
        "weights": w,
        "metrics": {"return": ret, "volatility": vol, "sharpe": sharpe},
        "optimizer_success": bool(res.success),
        "message": str(res.message),
    }


def efficient_frontier(returns, risk_free=0.0, max_weight=0.5, min_weight=0.0, points=30):
    r = returns.dropna()
    mu, cov = _annual_stats(r)
    n = len(r.columns)
    min_r, max_r = float(mu.min()), float(mu.max())
    targets = np.linspace(min_r, max_r, points)
    out = []

    for target in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        res = minimize(
            lambda w: np.sqrt(max(w @ cov @ w, 0)),
            np.repeat(1/n, n),
            method="SLSQP",
            bounds=[(min_weight, max_weight)] * n,
            constraints=cons,
        )
        if res.success:
            ret, vol, sh = _portfolio_stats(res.x, mu, cov, risk_free)
            out.append({"return": ret, "volatility": vol, "sharpe": sh})

    return pd.DataFrame(out)


def risk_contribution(returns, weights):
    cov = returns.cov() * TRADING_DAYS
    w = np.asarray(weights)
    port_vol = np.sqrt(max(w @ cov.values @ w, 1e-12))
    marginal = cov.values @ w / port_vol
    rc = w * marginal
    rc = rc / rc.sum()
    return pd.Series(rc, index=returns.columns).sort_values(ascending=False)


def diversification_stats(weights):
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    entropy = -np.sum(w * np.log(w))
    normalized = entropy / np.log(len(w)) if len(w) > 1 else 0.0
    hhi = np.sum(w ** 2)
    effective_n = 1 / hhi if hhi > 0 else 0
    return {
        "entropy": entropy,
        "normalized_entropy": normalized,
        "hhi": hhi,
        "effective_number_of_positions": effective_n,
    }


def sentiment_from_news(symbols):
    # Lightweight prototype. Replace with FinBERT / a finance-specific NLP model in production.
    positive = {
        "beat", "beats", "growth", "upgrade", "strong", "profit", "surge",
        "record", "bullish", "outperform", "positive", "improves", "buy",
    }
    negative = {
        "miss", "misses", "downgrade", "weak", "loss", "fall", "falls",
        "drop", "bearish", "underperform", "negative", "risk", "fraud",
    }

    rows = []
    for s in symbols:
        try:
            news = yf.Ticker(s).news or []
        except Exception:
            news = []

        scores = []
        titles = []
        for item in news[:15]:
            title = item.get("title", "") if isinstance(item, dict) else ""
            words = set(str(title).lower().replace(",", " ").replace(".", " ").split())
            score = (len(words & positive) - len(words & negative)) / max(len(words), 1)
            scores.append(score)
            titles.append(title)

        rows.append({
            "symbol": s,
            "sentiment": float(np.mean(scores)) if scores else np.nan,
            "news_count": len(titles),
            "latest_headline": titles[0] if titles else "",
        })
    return pd.DataFrame(rows)


def benchmark_symbols():
    return {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTY BANK": "^NSEBANK",
        "NIFTY 100": "^CNX100",
        "NIFTY 500": "^CRSLDX",
        "NIFTY IT": "^CNXIT",
        "NIFTY MIDCAP 50": "^NSEMDCP50",
    }


def format_pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "N/A"
