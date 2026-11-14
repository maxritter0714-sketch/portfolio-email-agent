"""Portfolio risk, momentum, and ETF look-through metrics."""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import pandas as pd
import yfinance as yf

import market
from analytics import TDAYS, buy_hold_returns, returns_on_calendar, risk_stats, to_weekly
from market import _to_eur_close

log = logging.getLogger(__name__)

DEFAULT_RISK_FREE = 0.035
MIN_HISTORY_DAYS = 60  # a holding with less history is left out of the risk figures


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


def _finite(value: object) -> float | None:
    """A plain float, or None for missing and NaN/inf values (shown as "n/a")."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _year_of_closes(symbol: str) -> pd.Series:
    """Last year of closes, from the price cache when the run already fetched them."""
    cached = market._hist_cache.get(symbol)
    if cached is not None:
        return cached
    return yf.Ticker(symbol).history(period="1y")["Close"]


def compute_advanced_metrics(positions: list[dict], fx_rates: dict[str, float]) -> dict:
    """Risk figures of today's holdings over the last year, from the dashboard's engine.

    The portfolio is today's holdings held unchanged (`analytics.buy_hold_returns`)
    and the figures come from `analytics.risk_stats`, so the email and the
    dashboard's 1Y column use the same formulas and the same benchmark ETFs.
    Beta, alpha and correlation are against the S&P 500 ETF. A value that
    cannot be computed is None, never a neutral-looking default.
    """
    risk_free = risk_free_rate()

    log.info("Computing advanced metrics…")

    total_val = sum(p["value_eur"] for p in positions)
    if total_val == 0:
        return {}

    prices: dict[str, pd.Series] = {}
    units: dict[str, float] = {}
    for p in positions:
        ticker = p["ticker"]
        try:
            close = _year_of_closes(ticker)
            if close is None or close.empty or len(close) < MIN_HISTORY_DAYS:
                continue
            eur = _to_eur_close(close, p.get("currency", "USD"), fx_rates).dropna()
        except Exception as exc:
            log.debug("No price history for %s in metrics: %s", ticker, exc)
            continue
        if eur.empty or not float(eur.iloc[-1]):
            continue
        prices[ticker] = eur
        # Share count implied by today's value, so one ticker held twice adds up.
        units[ticker] = units.get(ticker, 0.0) + p["value_eur"] / float(eur.iloc[-1])

    if not prices:
        log.warning("Not enough history for metrics")
        return {}

    px = pd.DataFrame(prices).sort_index()
    port_ret, _coverage = buy_hold_returns(px, pd.Series(units))

    try:
        bench = pd.DataFrame({
            key: _to_eur_close(_year_of_closes(market.BENCHMARK_ETFS[key][0]), "EUR", fx_rates)
            for key in ("sp500", "world")
        })
    except Exception as exc:
        log.error("Failed to fetch benchmark ETFs for metrics: %s", exc)
        return {}
    bench_ret = returns_on_calendar(bench, px.index).reindex(port_ret.index).fillna(0.0)

    stats = risk_stats(port_ret, bench_ret["sp500"], bench_ret["world"], risk_free)
    if not stats:
        log.warning("Not enough history for metrics")
        return {}

    pr = port_ret.to_numpy(dtype=float)
    weekly = to_weekly(pd.DataFrame({"p": port_ret, "s": bench_ret["sp500"]})).dropna()
    pr_w = weekly["p"].to_numpy(dtype=float)
    sr_w = weekly["s"].to_numpy(dtype=float)
    enough_weeks = len(pr_w) >= 2

    hhi = sum((p["value_eur"] / total_val) ** 2 for p in positions)
    win_rate = sum(1 for p in positions if p["pl_pct"] > 0) / len(positions) * 100

    metrics = dict(
        sharpe=stats["sharpe"], sortino=stats["sortino"], beta=stats["beta"], alpha=stats["alpha"],
        volatility=stats["volatility"], max_drawdown=stats["max_dd"], calmar=stats["calmar"],
        var_95=stats["var_95"], correlation=stats["correlation"], hhi=hhi, win_rate=win_rate,
        cvar_95=stats["cvar_95"],
        sharpe_90d=_trailing_sharpe(pr, 90, risk_free, TDAYS),
        beta_13w=_trailing_beta(pr_w, sr_w, 13) if enough_weeks else None,
        beta_26w=_trailing_beta(pr_w, sr_w, 26) if enough_weeks else None,
        vol_30d=_trailing_vol(pr, 30, TDAYS),
    )
    metrics = {key: _finite(value) for key, value in metrics.items()}
    log.info("Metrics: %s", ", ".join(
        f"{key}={metrics[key]:.2f}" for key in ("sharpe", "beta", "max_drawdown", "var_95")
        if metrics[key] is not None
    ))
    return metrics


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
