"""Reads the personal files in data/ and decides which Saturdays are send days."""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

from paths import DATA_DIR

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"ticker", "name", "shares", "avg_buy_price"}

# First send Saturday (issue № 1). Later sends are every 14 days from this date
# so the gap stays a true fortnight — ISO even-week would stretch to 21 days
# in years that have ISO week 53.
_SEND_EPOCH = date(2025, 1, 11)


def is_send_saturday(today: date | None = None) -> bool:
    """True on every other Saturday, starting 11 Jan 2025 (issue № 1)."""
    today = today or date.today()
    if today.weekday() != 5:
        return False
    if today < _SEND_EPOCH:
        return False
    return (today - _SEND_EPOCH).days % 14 == 0


def issue_number(today: date | None = None) -> int:
    """Fortnightly issue count, with 11 Jan 2025 as № 1."""
    today = today or date.today()
    return max(1, (today - _SEND_EPOCH).days // 14 + 1)


def load_portfolio(filepath: str | Path = DATA_DIR / "portfolio.csv") -> list[dict]:
    if not os.path.exists(filepath):
        log.error("portfolio.csv not found at '%s'", filepath)
        print(
            f"Error: '{filepath}' does not exist.\n"
            "Create it with columns: ticker, name, shares, avg_buy_price"
        )
        sys.exit(1)

    try:
        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            columns = set(reader.fieldnames or [])
    except Exception as exc:
        log.error("Failed to read %s: %s", filepath, exc)
        print(f"Error reading '{filepath}': {exc}")
        sys.exit(1)

    missing = REQUIRED_COLUMNS - columns
    if missing:
        log.error("Missing columns in portfolio.csv: %s", missing)
        print(
            f"Error: portfolio.csv is missing required columns: {', '.join(sorted(missing))}\n"
            f"Required: ticker, name, shares, avg_buy_price"
        )
        sys.exit(1)

    if not rows:
        print("Error: portfolio.csv has no data rows.")
        sys.exit(1)

    log.info("Portfolio loaded: %d holdings from '%s'", len(rows), filepath)
    return rows


_EXPORT_PATH = DATA_DIR / "finanzfluss_export.csv"
_UNMAPPED_PATH = DATA_DIR / "unmapped_isins.json"


def holdings_as_of(portfolio_path: str | Path = DATA_DIR / "portfolio.csv") -> date | None:
    """Date the holdings were last exported.

    In GitHub Actions the workflow sets PORTFOLIO_AS_OF from the data repo's
    last commit of the export (file mtimes there are just checkout time).
    Locally, the export's file date is used, else portfolio.csv's.
    """
    raw = os.getenv("PORTFOLIO_AS_OF", "").strip()
    if raw:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            log.warning("Invalid PORTFOLIO_AS_OF=%r — falling back to file date", raw)
    for path in (_EXPORT_PATH, Path(portfolio_path)):
        if path.exists():
            return date.fromtimestamp(path.stat().st_mtime)
    return None


def load_unmapped_isins(path: Path = _UNMAPPED_PATH) -> list[dict]:
    """Holdings import_portfolio.py could not map to a ticker (left out of the report)."""
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read %s (%s)", path, exc)
        return []
    return rows if isinstance(rows, list) else []


def _is_float(value: object | None) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, bytearray, int, float, np.integer, np.floating)):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    return False


_TARGETS_PATH = DATA_DIR / "targets.json"


def load_targets(path: Path = _TARGETS_PATH) -> dict | None:
    """Optional target allocation from targets.json, or None when not configured."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        log.warning("Could not read %s (%s) — skipping target allocation", path, exc)
        return None


_CASHFLOWS_PATH = DATA_DIR / "cashflows.csv"


def load_cashflows(path: Path = _CASHFLOWS_PATH) -> list[dict]:
    """Daily invested/income totals written by import_cashflows.py, or [] if absent.

    `orders` (one-off orders that day) is None for files written before that
    column existed.
    """
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return [
        dict(date=r["date"], invested_eur=float(r["invested_eur"]), income_eur=float(r["income_eur"]),
             orders=int(r["orders"]) if (r.get("orders") or "").strip().isdigit() else None)
        for r in rows
        if r.get("date") and _is_float(r.get("invested_eur")) and _is_float(r.get("income_eur"))
    ]
