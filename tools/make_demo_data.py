"""Build a synthetic demo portfolio in docs/demo/ for README screenshots.

The holdings are well-known stocks and ETFs with made-up share counts; every
price comes from Yahoo Finance, so the charts move like a real portfolio
without exposing anyone's actual positions or cash flows.

Usage:
    python tools/make_demo_data.py
    PORTFOLIO_DATA_DIR=docs/demo python agent.py --force --preview
    PORTFOLIO_DATA_DIR=docs/demo python dashboard.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from market import _to_eur_close, get_fx_rates, is_send_saturday, issue_number  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent.parent / "docs" / "demo"

# ticker, name, currency, shares
HOLDINGS = [
    ("SXR8.DE", "iShares Core S&P 500 UCITS ETF", "EUR", 14),
    ("EUNL.DE", "iShares Core MSCI World UCITS ETF", "EUR", 55),
    ("IS3N.DE", "iShares Core MSCI EM IMI UCITS ETF", "EUR", 120),
    ("AAPL", "Apple", "USD", 18),
    ("MSFT", "Microsoft", "USD", 9),
    ("NVDA", "NVIDIA", "USD", 30),
    ("AMZN", "Amazon", "USD", 16),
    ("GOOGL", "Alphabet", "USD", 14),
    ("ASML.AS", "ASML Holding", "EUR", 4),
    ("SAP.DE", "SAP", "EUR", 10),
    ("ALV.DE", "Allianz", "EUR", 7),
    ("TSM", "Taiwan Semiconductor", "USD", 12),
]
MONTHLY_SAVINGS_EUR = 600.0
PLAN_START = date(2023, 1, 2)
HISTORY_DAYS = 120  # trading-day rows in history.csv


def _eur_closes(fx_rates: dict[str, float]) -> pd.DataFrame:
    tickers = [h[0] for h in HOLDINGS] + ["^GSPC"]
    raw = yf.download(tickers, period="3y", auto_adjust=True, progress=False)["Close"]
    out = {}
    for ticker, _, currency, _ in HOLDINGS:
        out[ticker] = _to_eur_close(raw[ticker].dropna(), currency, fx_rates, fx_period="3y")
    out["^GSPC"] = raw["^GSPC"].dropna()
    out["^GSPC_EUR"] = _to_eur_close(raw["^GSPC"].dropna(), "USD", fx_rates, fx_period="3y")
    return pd.DataFrame(out).ffill().dropna()


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _portfolio(px: pd.DataFrame) -> float:
    """Average buy price = mean EUR close since the savings plan began (a fair DCA proxy)."""
    since = px[px.index >= pd.Timestamp(PLAN_START)]
    rows, cost = [], 0.0
    for ticker, name, _, shares in HOLDINGS:
        avg = float(since[ticker].mean())
        cost += shares * avg
        rows.append([ticker, name, shares, round(avg, 2)])
    _write_csv(DEMO_DIR / "portfolio.csv", ["ticker", "name", "shares", "avg_buy_price"], rows)
    return cost


def _cashflows(cost: float, today: date) -> None:
    """A monthly savings plan whose deposits add up to the cost basis, plus a few dividends."""
    months = pd.date_range(PLAN_START, today, freq="MS")
    per_month = round(cost / len(months), 2)
    rows = []
    for i, m in enumerate(months):
        day = m.date() + timedelta(days=1)
        income = round(per_month * 0.04, 2) if m.month in (3, 6, 9, 12) else 0.0
        orders = 1 if i % 5 == 0 else 0  # an occasional one-off order on top of the plan
        rows.append([day.isoformat(), per_month, income, orders])
    _write_csv(DEMO_DIR / "cashflows.csv", ["date", "invested_eur", "income_eur", "orders"], rows)


def _history(px: pd.DataFrame, today: date) -> None:
    recent = px[px.index < pd.Timestamp(today)].tail(HISTORY_DAYS)  # closed sessions only
    shares = pd.Series({h[0]: h[3] for h in HOLDINGS})
    value = (recent[shares.index] * shares).sum(axis=1)
    port_index = 100 * value / value.iloc[0]
    rows = []
    for ts in recent.index:
        d = ts.date()
        rows.append([
            d.isoformat(), issue_number(d), round(float(value[ts]), 2),
            round(float(recent.at[ts, "^GSPC"]), 4), f"{d.isoformat()}T22:30:00", "scheduled",
            round(float(port_index[ts]), 4), round(float(recent.at[ts, "^GSPC_EUR"]), 4),
        ])
    _write_csv(DEMO_DIR / "history.csv",
               ["date", "issue", "total_value_eur", "sp500_close", "datetime", "source",
                "port_index", "sp500_eur"], rows)


# (reports ago, ticker or "", action, title)
_RECS = [
    (13, "NVDA", "TRIM", "NVIDIA"),
    (13, "", "RESEARCH", "Check overlap between S&P 500 and MSCI World ETFs"),
    (11, "IS3N.DE", "ADD", "Emerging markets ETF"),
    (9, "TSM", "TRIM", "Taiwan Semiconductor"),
    (7, "ALV.DE", "ADD", "Allianz"),
    (5, "AAPL", "TRIM", "Apple"),
    (3, "SAP.DE", "EXIT", "SAP"),
    (2, "EUNL.DE", "ADD", "MSCI World ETF"),
    (1, "NVDA", "TRIM", "NVIDIA"),
    (1, "", "RESEARCH", "Rebalance towards broad ETFs"),
]


def _recommendations(px: pd.DataFrame, today: date) -> None:
    """Sample advice dated on past report Saturdays, priced at that day's EUR close."""
    last_send = today - timedelta(days=1)
    while not is_send_saturday(last_send):
        last_send -= timedelta(days=1)
    rows, conviction = [], {}
    for ago, ticker, action, title in _RECS:
        d = last_send - timedelta(weeks=2 * (ago - 1))
        issue = issue_number(d)
        price = ""
        if ticker:
            price = round(float(px[ticker][px.index <= pd.Timestamp(d)].iloc[-1]), 2)
            conviction[ticker] = dict(action=action, report=issue, date=d.isoformat(), price_eur=price)
        rows.append([d.isoformat(), issue, ticker, action, price, title])
    _write_csv(DEMO_DIR / "recommendations.csv",
               ["date", "report", "ticker", "action", "price_eur", "title"], rows)
    (DEMO_DIR / "conviction.json").write_text(json.dumps(conviction, indent=2), encoding="utf-8")


def _targets() -> None:
    targets = {
        "tolerance_pct": 5,
        "groups": [
            {"name": "Core ETFs", "target_pct": 60, "tickers": ["SXR8.DE", "EUNL.DE", "IS3N.DE"]},
            {"name": "Single stocks", "target_pct": 40,
             "tickers": [h[0] for h in HOLDINGS if not h[1].startswith("iShares")]},
        ],
    }
    (DEMO_DIR / "targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    px = _eur_closes(get_fx_rates())
    cost = _portfolio(px)
    _cashflows(cost, today)
    _history(px, today)
    _recommendations(px, today)
    _targets()
    print(f"Demo data written to {DEMO_DIR} (cost basis {cost:,.0f} EUR)")


if __name__ == "__main__":
    main()
