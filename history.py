"""data/history.csv: one row per trading day with the portfolio's value and index."""
from __future__ import annotations

import csv
import logging
from datetime import date
from datetime import datetime as dt
from typing import Any

import pandas as pd

import market
from market import SP500_INDEX, _to_eur_close, fx_fallback_currencies
from paths import DATA_DIR
from portfolio_io import _is_float

log = logging.getLogger(__name__)


_HISTORY_PATH = DATA_DIR / "history.csv"
_HISTORY_COLS = (
    "date", "issue", "total_value_eur", "sp500_close", "datetime", "source",
    "port_index", "sp500_eur",
)


def _read_history() -> list[dict]:
    if not _HISTORY_PATH.exists():
        return []
    with open(_HISTORY_PATH, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _last_indexed_row(rows: list[dict]) -> dict | None:
    """Most recent scheduled row that carries a portfolio index value."""
    indexed = [
        r for r in rows
        if r.get("source", "scheduled") == "scheduled"
        and r.get("date")
        and _is_float(r.get("port_index"))
    ]
    return indexed[-1] if indexed else None


def next_portfolio_index(
    positions: list[dict], fx_rates: dict[str, float], today: date | None = None,
) -> float:
    """Chain the portfolio's EUR price return since the last indexed history row.

    Deposits and withdrawals don't move the index: the return is measured on
    the current holdings' share counts, held constant over the period. Trades
    inside the period are therefore only approximated. Starts at 100.
    """
    today = today or date.today()
    last = _last_indexed_row(_read_history())
    if last is None:
        return 100.0
    last_index = float(last["port_index"])
    since = pd.Timestamp(last["date"])
    if since.date() >= today:
        return last_index

    start_value = end_value = 0.0
    for p in positions:
        close = market.cached_closes(p["ticker"])
        if close is None or close.empty:
            continue
        eur = _to_eur_close(close, p.get("currency", "USD"), fx_rates).dropna()
        before = eur[eur.index <= since]
        if before.empty:
            continue
        start_value += p["shares"] * float(before.iloc[-1])
        end_value += p["shares"] * float(eur.iloc[-1])

    if not start_value:
        return last_index
    return last_index * end_value / start_value


def append_history(
    summary: dict,
    issue_num: int,
    source: str = "scheduled",
    port_index: float | None = None,
    sp500_eur: float | None = None,
    day: date | None = None,
) -> bool:
    """Append one row to history.csv per (date, source) pair; returns whether a row was written.

    `day` is the trading day the prices belong to (default: today), so a
    weekend run after the Friday snapshot finds Friday's row and skips.
    """
    today_str = (day or date.today()).isoformat()
    now_str = dt.now().isoformat(timespec="seconds")
    existing: list[dict] = []

    if _HISTORY_PATH.exists():
        has_bom = _HISTORY_PATH.read_bytes()[:3] == b"\xef\xbb\xbf"
        with open(_HISTORY_PATH, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            existing = list(reader)
            file_cols = [c.strip() for c in (reader.fieldnames or [])]

        missing_cols = set(_HISTORY_COLS) - set(file_cols)
        if has_bom or missing_cols:
            # Migrate: rewrite with full schema; backfill old rows with sensible defaults.
            for row in existing:
                row.setdefault("datetime", row.get("date", ""))
                row.setdefault("source", "scheduled")
            with open(_HISTORY_PATH, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(_HISTORY_COLS), extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing)
            log.info("History: migrated schema (added: %s, bom=%s).", missing_cols, has_bom)

        if any(r.get("date") == today_str and r.get("source", "scheduled") == source for r in existing):
            log.info("History: %s entry for %s already exists — skipping.", source, today_str)
            return False

    sp500_series = market.cached_closes(SP500_INDEX)
    if sp500_series is None or sp500_series.empty:
        log.warning("S&P 500 not in cache — history row will have sp500_close=0")
        sp500_close = 0.0
    else:
        sp500_close = float(sp500_series.iloc[-1])

    row: dict[str, Any] = {
        "date": today_str,
        "issue": issue_num,
        "total_value_eur": round(summary["total_value_eur"], 2),
        "sp500_close": round(sp500_close, 4),
        "datetime": now_str,
        "source": source,
        "port_index": round(port_index, 4) if port_index is not None else "",
        "sp500_eur": round(sp500_eur, 4) if sp500_eur is not None else "",
    }
    write_header = not _HISTORY_PATH.exists()
    with open(_HISTORY_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_HISTORY_COLS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return True


def history_blockers(
    positions: list[dict],
    skipped: list[dict],
    unknown_fx: list[dict],
    sp500_close: pd.Series | None,
) -> list[str]:
    """Reasons this run's data must not become a history row; empty when it is complete.

    A row built on a missing holding, a guessed currency or a constant FX rate
    looks like any other row but carries a wrong total, so none is written.
    """
    reasons: list[str] = []
    if skipped:
        reasons.append("no price for " + ", ".join(s["ticker"] for s in skipped))
    if unknown_fx:
        reasons.append("no FX rate for " + ", ".join(
            f"{s['ticker']} ({s['currency']})" for s in unknown_fx))
    held = {"GBP" if p.get("currency") in ("GBp", "GBX") else p.get("currency") for p in positions}
    constant = sorted(fx_fallback_currencies() & held)
    if constant:
        reasons.append("FX rate could not be fetched for " + ", ".join(constant))
    if sp500_close is None or sp500_close.empty:
        reasons.append("no S&P 500 close to date the row")
    return reasons


def last_trading_day(close: pd.Series | None, fallback: date | None = None) -> date:
    """Date of the last close in `close` (the S&P 500), else `fallback` or today.

    History rows are dated by the trading day their prices come from, not the
    run date: the Saturday run then reuses Friday's date, and a US holiday
    resolves to the previous session.
    """
    fallback = fallback or date.today()
    if close is None or close.empty:
        return fallback
    last = pd.Timestamp(close.index[-1]).date()
    return min(last, fallback)


def value_change_breakdown(
    current_value: float,
    cashflows: list[dict],
    history: list[dict] | None = None,
    today: date | None = None,
) -> dict | None:
    """Split the value change since a baseline into money paid in vs market gain.

    The baseline is the first history row of the current year, else the first
    row overall. Dividends and interest land on the cash account, so they are
    reported separately and are not part of the value change. None when there
    is no baseline or no cash-flow data.
    """
    if not cashflows:
        return None
    today = today or date.today()
    rows = [
        r for r in (history if history is not None else _read_history())
        if r.get("date") and _is_float(r.get("total_value_eur"))
    ]
    if not rows:
        return None
    this_year = [r for r in rows if r["date"].startswith(str(today.year))]
    base = (this_year or rows)[0]
    since = base["date"]
    period = [c for c in cashflows if since < c["date"] <= today.isoformat()]
    start_value = float(base["total_value_eur"])
    invested = sum(c["invested_eur"] for c in period)
    change = current_value - start_value
    return dict(
        since=date.fromisoformat(since),
        start_value=start_value,
        change=change,
        invested=invested,
        market=change - invested,
        income=sum(c["income_eur"] for c in period),
        flows_as_of=date.fromisoformat(cashflows[-1]["date"]),
    )


def get_history_ytd(current_value: float) -> float | None:
    """Book-value change vs the first January snapshot in history.csv, or None.

    Includes deposits and withdrawals. Not a time-weighted return.
    """
    if not _HISTORY_PATH.exists():
        return None
    current_year = date.today().year
    with open(_HISTORY_PATH, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    jan_rows = [
        r for r in rows
        if r.get("date", "").startswith(f"{current_year}-01")
        and _is_float(r.get("total_value_eur"))
    ]
    if not jan_rows:
        return None
    jan_value = float(jan_rows[0]["total_value_eur"])
    return (current_value - jan_value) / jan_value * 100 if jan_value else None
