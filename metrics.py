"""Portfolio risk, momentum, and ETF look-through metrics."""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

import market
from market import _to_eur_close

log = logging.getLogger(__name__)

DEFAULT_RISK_FREE = 0.035


def risk_free_rate() -> float:
    """Annual risk-free rate for Sharpe/Sortino/alpha. Override with RISK_FREE_RATE."""
    raw = os.getenv("RISK_FREE_RATE", "")
    if not raw.strip():
        return DEFAULT_RISK_FREE
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid RISK_FREE_RATE=%r — using %.3f", raw, DEFAULT_RISK_FREE)
        return DEFAULT_RISK_FREE


def _trailing(arr: np.ndarray, window: int) -> np.ndarray:
    """Last `window` entries, or the full array if shorter.

    Falls back to the full period when history is shorter than the window
    (e.g. a 75-day-old position asked for a 90d slice) so callers degrade to
    the same figure already shown elsewhere as the full-period metric,
    instead of crashing or returning a misleading dummy value.
    """
    return arr[-window:] if len(arr) >= window else arr


def _trailing_sharpe(pr: np.ndarray, window: int, risk_free: float, tdays: int) -> float:
    w = _trailing(pr, window)
    ann_ret = np.mean(w) * tdays
    ann_vol = np.std(w, ddof=1) * np.sqrt(tdays)
    return (ann_ret - risk_free) / ann_vol if ann_vol else 0.0


def _trailing_beta(pr: np.ndarray, sr: np.ndarray, window: int) -> float | None:
    """None when the benchmark window has zero variance (degenerate window).

    A fallback of 1.0 ("moves exactly like the market") would be a specific,
    potentially false claim rather than a neutral default — better to signal
    "no data" and have the caller omit it than risk Claude commenting on a
    fabricated stable beta.
    """
    wp, wb = _trailing(pr, window), _trailing(sr, window)
    cov = np.cov(wp, wb)
    if cov[1, 1] == 0:
        return None
    return float(cov[0, 1] / cov[1, 1])


def _trailing_vol(pr: np.ndarray, window: int, tdays: int) -> float:
    return np.std(_trailing(pr, window), ddof=1) * np.sqrt(tdays)


def compute_advanced_metrics(positions: list[dict], fx_rates: dict[str, float]) -> dict:
    RISK_FREE = risk_free_rate()
    TDAYS = 252

    log.info("Computing advanced metrics…")

    total_val = sum(p["value_eur"] for p in positions)
    if total_val == 0:
        return {}

    ret_df: dict[str, object] = {}
    weights_map: dict[str, float] = {}
    for p in positions:
        try:
            hist_close = market._hist_cache.get(p["ticker"])
            if hist_close is None:
                hist_close = yf.Ticker(p["ticker"]).history(period="1y")["Close"]
            if hist_close.empty or len(hist_close) < 60:
                continue
            hist_eur = _to_eur_close(hist_close, p.get("currency", "USD"), fx_rates)
            ret_df[p["ticker"]] = hist_eur.pct_change().dropna()
            weights_map[p["ticker"]] = p["value_eur"] / total_val
        except Exception:
            continue

    if not ret_df:
        log.warning("Not enough history for metrics")
        return {}

    df = pd.DataFrame(ret_df)
    w = pd.Series(weights_map).reindex(df.columns).fillna(0)
    # Renormalise weights per day: tickers absent (holiday/NaN) are excluded
    # so the portfolio return is not understated.
    active_w = df.notna().mul(w, axis=1)
    day_w_sum = active_w.sum(axis=1).replace(0, np.nan)
    port_ret_series = df.fillna(0).mul(active_w).sum(axis=1).div(day_w_sum)
    port_ret_series = port_ret_series.dropna()

    if len(port_ret_series) < 60:
        log.warning("Not enough history for metrics")
        return {}

    try:
        _cached = market._hist_cache.get("^GSPC")
        sp500_close = _cached if _cached is not None else yf.Ticker("^GSPC").history(period="1y")["Close"]
        sp500_eur = _to_eur_close(sp500_close, "USD", fx_rates)
        sp500_ret = sp500_eur.pct_change().dropna()
    except Exception as exc:
        log.error("Failed to fetch S&P 500 for metrics: %s", exc)
        return {}

    common_idx = port_ret_series.index.intersection(sp500_ret.index)
    if len(common_idx) < 60:
        return {}

    pr = np.asarray(port_ret_series.loc[common_idx].values, dtype=float)
    sr = np.asarray(sp500_ret.loc[common_idx].values, dtype=float)

    ann_ret = np.mean(pr) * TDAYS
    ann_vol = np.std(pr, ddof=1) * np.sqrt(TDAYS)
    sp500_ann = np.mean(sr) * TDAYS

    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol else 0.0

    neg = pr[pr < 0]
    down_vol = np.std(neg, ddof=1) * np.sqrt(TDAYS) if len(neg) > 1 else ann_vol
    sortino = (ann_ret - RISK_FREE) / down_vol if down_vol else 0.0

    cov = np.cov(pr, sr)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] else 1.0
    alpha = ann_ret - (RISK_FREE + beta * (sp500_ann - RISK_FREE))

    cum = np.cumprod(1 + pr)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - peak) / peak))

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    var_95 = float(np.percentile(pr, 5))
    # Full-period (not rolling) Expected Shortfall: mean return in the worst 5% of all-time daily returns.
    cvar_95 = float(np.mean(pr[pr <= np.percentile(pr, 5)]))
    correlation = float(np.corrcoef(pr, sr)[0, 1])

    sharpe_90d = _trailing_sharpe(pr, 90, RISK_FREE, TDAYS)
    beta_30d = _trailing_beta(pr, sr, 30)
    beta_90d = _trailing_beta(pr, sr, 90)
    vol_30d = _trailing_vol(pr, 30, TDAYS)

    hhi = sum((p["value_eur"] / total_val) ** 2 for p in positions) if total_val else 0.0
    win_rate = sum(1 for p in positions if p["pl_pct"] > 0) / len(positions) * 100

    log.info(
        "Metrics: Sharpe=%.2f  Beta=%.2f  MaxDD=%.1f%%  VaR95=%.2f%%",
        sharpe, beta, max_dd * 100, var_95 * 100,
    )
    return dict(
        sharpe=sharpe, sortino=sortino, beta=beta, alpha=alpha,
        volatility=ann_vol, max_drawdown=max_dd, calmar=calmar,
        var_95=var_95, correlation=correlation, hhi=hhi, win_rate=win_rate,
        cvar_95=cvar_95, sharpe_90d=sharpe_90d,
        beta_30d=beta_30d, beta_90d=beta_90d, vol_30d=vol_30d,
    )


def compute_momentum_scores(positions: list[dict]) -> list[dict]:
    """Multi-timeframe momentum score per position, ranked into quartiles.

    r1m is computed and reported but excluded from the score: academic
    momentum research ("12-2 momentum", Jegadeesh-Titman) deliberately skips
    the most recent month, since 1-month returns tend to mean-revert rather
    than continue, unlike 3-12 month returns.
    """
    scored: list[dict] = []
    for p in positions:
        close = market._hist_cache.get(p["ticker"])
        if close is None or len(close) < 63:
            continue
        vals = close.to_numpy(dtype=float)
        current = vals[-1]
        r1m = current / vals[-21] - 1
        r3m = current / vals[-63] - 1
        r6m = current / vals[-126] - 1 if len(vals) >= 126 else r3m
        r12m = current / vals[0] - 1
        score = 0.35 * r3m + 0.35 * r6m + 0.30 * r12m
        scored.append(dict(
            ticker=p["ticker"], name=p["name"], score=score,
            r1m=r1m, r3m=r3m, r6m=r6m, r12m=r12m,
        ))

    scored.sort(key=lambda d: d["score"])
    n = len(scored)
    for i, d in enumerate(scored):
        d["quartile"] = min(4, i * 4 // n + 1)

    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored


def compute_lookthrough_exposure(positions: list[dict]) -> list[dict]:
    """Effective per-ticker exposure including ETF look-through.

    Returns two kinds of entries, both presented as neutral fact rather
    than a risk signal — more or hidden exposure to a given name is not
    inherently good or bad, that depends on the reader's own view of it:
      - "amplified": a directly-held position whose true effective exposure
        (direct + embedded in ETFs) is at least 20% higher than the direct
        position alone shows.
      - "hidden": a stock not held directly at all, but with non-trivial
        exposure (>=0.3% of total portfolio value) purely through ETFs.

    ETF `top_holdings` only covers the largest ~10 constituents per fund,
    so this is a conservative floor on both kinds, not exhaustive.
    """
    total_value = sum(p["value_eur"] for p in positions)
    if not total_value:
        return []

    direct: dict[str, float] = {}
    effective: dict[str, float] = {}
    names: dict[str, str] = {}

    for p in positions:
        direct[p["ticker"]] = direct.get(p["ticker"], 0.0) + p["value_eur"]
        effective[p["ticker"]] = effective.get(p["ticker"], 0.0) + p["value_eur"]
        names[p["ticker"]] = p["name"]

    for p in positions:
        for constituent, weight in p.get("holdings_weights", {}).items():
            effective[constituent] = effective.get(constituent, 0.0) + p["value_eur"] * weight
            names.setdefault(constituent, constituent)

    results = []
    for ticker, effective_eur in effective.items():
        direct_eur = direct.get(ticker, 0.0)
        hidden_eur = effective_eur - direct_eur
        hidden_pct = hidden_eur / total_value * 100
        if direct_eur > 0:
            if hidden_eur / direct_eur < 0.20:
                continue
            kind = "amplified"
        else:
            if hidden_pct < 0.3:
                continue
            kind = "hidden"
        results.append(dict(
            ticker=ticker, name=names.get(ticker, ticker), kind=kind,
            direct_eur=direct_eur, effective_eur=effective_eur,
            hidden_eur=hidden_eur, hidden_pct=hidden_pct,
        ))

    results.sort(key=lambda d: d["hidden_eur"], reverse=True)
    return results
