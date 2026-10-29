#!/usr/bin/env python3
"""
dashboard.py

Builds an interactive performance dashboard as one self-contained HTML file.
Fetches prices only: no Claude, no news, no email, and nothing in data/ is
written.

Usage:
    python dashboard.py            # writes output/dashboard.html
    python dashboard.py --open     # ...and opens it in the browser
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
import webbrowser
from datetime import date, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

import analytics as an
from market import (
    _read_history,
    _to_eur_close,
    fetch_portfolio_data,
    get_fx_rates,
    holdings_as_of,
    lifetime_return,
    load_cashflows,
    load_portfolio,
    region_exposure,
    sector_exposure,
)
from metrics import compute_momentum_scores, risk_free_rate
from paths import OUTPUT_DIR
from theme import (
    BG_CHART, BG_CREAM, BURGUNDY, CHARCOAL, DARK_GOLD, GOLD, NAVY, SAGE,
    TEXT_PRIMARY, TEXT_SECONDARY, _PALETTE,
)

log = logging.getLogger(__name__)

HISTORY_PERIOD = "7y"
# Total-return benchmarks, so they compare fairly with dividend-adjusted holdings.
BENCHMARKS = {
    "sp500": ("^SP500TR", "S&P 500 TR"),
    "world": ("URTH", "MSCI World"),
}
# The backtest starts once this share of today's value was already listed.
MIN_COVERAGE = 0.5
POSITION_PERIODS = ("1M", "3M", "YTD", "1Y")
RISK_WINDOWS = ("1Y", "3Y", "5Y")
TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"
OUTPUT_PATH = OUTPUT_DIR / "dashboard.html"


# ── Data fetching ────────────────────────────────────────────────────────────

def fetch_eur_prices(currencies: dict[str, str], fx_rates: dict[str, float]) -> pd.DataFrame:
    """Dividend-adjusted daily closes in EUR (dates × tickers) over HISTORY_PERIOD."""
    tickers = list(currencies)
    raw = yf.download(tickers, period=HISTORY_PERIOD, auto_adjust=True,
                      progress=False, threads=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    cols = {}
    for ticker, currency in currencies.items():
        if ticker not in raw.columns:
            log.warning("No long history for %s", ticker)
            continue
        close = raw[ticker].dropna()
        if close.empty:
            log.warning("No long history for %s", ticker)
            continue
        cols[ticker] = _to_eur_close(close, currency, fx_rates, fx_period=HISTORY_PERIOD)
    return pd.DataFrame(cols).sort_index()


# ── Payload ──────────────────────────────────────────────────────────────────

def _series(s: pd.Series) -> dict:
    return dict(dates=[d.date().isoformat() for d in s.index], values=s.tolist())


def _real_record(positions_value: float, cashflows: list[dict], history: list[dict],
                 today: date) -> dict:
    """Actual money paid in vs value, from cashflows.csv and history.csv."""
    snapshots = [
        dict(date=r["date"][:10], value=float(r["total_value_eur"]))
        for r in history
        if r.get("date") and r.get("total_value_eur")
    ]
    snapshots.append(dict(date=today.isoformat(), value=positions_value))
    lifetime = lifetime_return(positions_value, cashflows)
    return dict(
        flows=an.cumulative_flows(cashflows),
        snapshots=snapshots,
        value=positions_value,
        invested=lifetime["invested"] if lifetime else None,
        gain=lifetime["gain"] if lifetime else None,
        gain_pct=lifetime["pct"] / 100 if lifetime else None,
        income=lifetime["income"] if lifetime else None,
        since=lifetime["since"].isoformat() if lifetime else None,
        xirr=an.portfolio_xirr(cashflows, positions_value, today) if cashflows else None,
        flows_as_of=cashflows[-1]["date"] if cashflows else None,
    )


def build_payload(
    positions: list[dict],
    prices: pd.DataFrame,
    bench_prices: pd.DataFrame,
    cashflows: list[dict],
    history: list[dict],
    risk_free: float,
    today: date,
) -> dict:
    total_value = sum(p["value_eur"] for p in positions)
    shares = pd.Series({p["ticker"]: p["shares"] for p in positions})
    weights = pd.Series({p["ticker"]: p["value_eur"] / total_value for p in positions})

    # Portfolio backtest from the first day enough of today's value was listed.
    port_ret, coverage = an.buy_hold_returns(prices, shares)
    covered = coverage[coverage >= MIN_COVERAGE]
    start = covered.index[0] if not covered.empty else port_ret.index[0]
    port_ret = port_ret[port_ret.index > start]
    coverage = coverage[coverage.index >= start]

    # Benchmarks on the portfolio calendar (carried over their holidays).
    bench = bench_prices.reindex(bench_prices.index.union(prices.index)).ffill().reindex(prices.index)
    bench_ret = bench.pct_change().reindex(port_ret.index).fillna(0.0)
    sp_ret, world_ret = bench_ret["sp500"], bench_ret["world"]

    indices = {
        "portfolio": an.index_from_returns(port_ret),
        "sp500": an.index_from_returns(sp_ret),
        "world": an.index_from_returns(world_ret),
    }
    end = port_ret.index[-1]

    risk = {}
    for key in RISK_WINDOWS:
        since = an.period_start(end, key)
        if since < port_ret.index[0]:
            continue
        mask = port_ret.index > since
        risk[key] = an.risk_stats(port_ret[mask], sp_ret[mask], world_ret[mask], risk_free)
    risk["Max"] = an.risk_stats(port_ret, sp_ret, world_ret, risk_free)

    # Per position: last year of each name's own returns.
    one_year = an.period_start(end, "1Y")
    returns_1y = pd.DataFrame({
        t: prices[t].dropna().pct_change().loc[lambda s: s.index > one_year]
        for t in prices.columns
    })
    pos_risk = an.position_risk(returns_1y, port_ret[port_ret.index > one_year],
                                sp_ret[sp_ret.index > one_year])
    risk_share, port_vol, div_ratio = an.risk_contributions(returns_1y, weights)
    contrib = {k: an.contributions(prices, shares, an.period_start(end, k)) for k in POSITION_PERIODS}
    momentum = {m["ticker"]: m for m in compute_momentum_scores(positions)}

    def cell(frame: pd.DataFrame, ticker: str, col: str):
        return frame.at[ticker, col] if ticker in frame.index and col in frame.columns else None

    rows = []
    for p in sorted(positions, key=lambda p: p["value_eur"], reverse=True):
        t = p["ticker"]
        rows.append(dict(
            ticker=t, name=p["name"], sector=p.get("sector"), region=p.get("region"),
            value=p["value_eur"], weight=weights[t], pl_pct=p["pl_pct"] / 100,
            returns={k: cell(contrib[k], t, "ret") for k in POSITION_PERIODS},
            contrib={k: cell(contrib[k], t, "contrib") for k in POSITION_PERIODS},
            volatility=cell(pos_risk, t, "volatility"),
            beta=cell(pos_risk, t, "beta"),
            max_dd=cell(pos_risk, t, "max_dd"),
            corr=cell(pos_risk, t, "corr_portfolio"),
            risk_share=float(risk_share[t]) if t in risk_share.index else None,
            momentum_q=momentum[t]["quartile"] if t in momentum else None,
            listed=prices[t].first_valid_index().date().isoformat() if t in prices.columns else None,
        ))

    ordered = [r["ticker"] for r in rows if r["ticker"] in returns_1y.columns]
    corr = an.to_weekly(returns_1y[ordered]).corr()

    def shares_of(exposure: dict[str, float]) -> list[dict]:
        return [dict(name=k, weight=v / total_value)
                for k, v in sorted(exposure.items(), key=lambda kv: kv[1], reverse=True)]

    return dict(
        generated=dt.now().isoformat(timespec="minutes"),
        holdings_as_of=(d.isoformat() if (d := holdings_as_of()) else None),
        risk_free=risk_free,
        benchmarks={k: label for k, (_, label) in BENCHMARKS.items()},
        real=_real_record(total_value, cashflows, history, today),
        series=dict(
            dates=[d.date().isoformat() for d in port_ret.index],
            portfolio=indices["portfolio"].tolist(),
            sp500=indices["sp500"].tolist(),
            world=indices["world"].tolist(),
            coverage=coverage.reindex(port_ret.index).ffill().tolist(),
        ),
        backtest_start=start.date().isoformat(),
        periods=an.period_table(indices, end),
        drawdown=an.drawdown_stats(indices["portfolio"]),
        monthly=an.monthly_returns(port_ret),
        risk=risk,
        rolling={k: _series(v) for k, v in an.rolling_stats(port_ret, sp_ret, risk_free).items()},
        positions=rows,
        structure=dict(
            **an.concentration(weights),
            volatility=port_vol,
            diversification_ratio=div_ratio,
            sectors=shares_of(sector_exposure(positions)),
            regions=shares_of(region_exposure(positions)),
        ),
        correlation=dict(tickers=ordered, matrix=corr.round(3).to_numpy().tolist()),
        theme=dict(
            bg=BG_CREAM, panel=BG_CHART, text=TEXT_PRIMARY, muted=TEXT_SECONDARY,
            gold=GOLD, dark_gold=DARK_GOLD, navy=NAVY, sage=SAGE, burgundy=BURGUNDY,
            charcoal=CHARCOAL, palette=_PALETTE,
        ),
    )


def _json_ready(obj):
    """NaN/inf → null, numpy scalars → Python, floats rounded to keep the file small."""
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else round(value, 6)
    return obj


def render_html(payload: dict, template: str) -> str:
    data = json.dumps(_json_ready(payload), separators=(",", ":"), allow_nan=False)
    # "</" inside the inline <script> would end it early.
    return template.replace("/*__DATA__*/null", data.replace("</", "<\\/"))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the interactive performance dashboard")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the browser")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    # yfinance under pandas 3.1 floods the console with inplace deprecations.
    warnings.filterwarnings("ignore", message=".*inplace keyword.*")

    rows = load_portfolio()
    fx_rates = get_fx_rates()
    positions, _summary, _unknown_fx, skipped = fetch_portfolio_data(rows, fx_rates)
    if skipped:
        log.warning("Not in dashboard (no price data): %s", ", ".join(s["ticker"] for s in skipped))

    log.info("Fetching %s of price history", HISTORY_PERIOD)
    prices = fetch_eur_prices({p["ticker"]: p.get("currency", "USD") for p in positions}, fx_rates)
    bench = fetch_eur_prices({sym: "USD" for sym, _ in BENCHMARKS.values()}, fx_rates)
    bench = bench.rename(columns={sym: key for key, (sym, _) in BENCHMARKS.items()})

    payload = build_payload(
        positions=[p for p in positions if p["ticker"] in prices.columns],
        prices=prices,
        bench_prices=bench,
        cashflows=load_cashflows(),
        history=_read_history(),
        risk_free=risk_free_rate(),
        today=date.today(),
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(render_html(payload, TEMPLATE_PATH.read_text(encoding="utf-8")),
                           encoding="utf-8")
    log.info("Dashboard written to %s", OUTPUT_PATH)
    print(f"Dashboard written to '{OUTPUT_PATH}'.")
    if args.open:
        webbrowser.open(OUTPUT_PATH.resolve().as_uri())


if __name__ == "__main__":
    main()
