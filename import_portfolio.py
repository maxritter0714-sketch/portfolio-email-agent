#!/usr/bin/env python3
"""
import_portfolio.py

Converts a portfolio-tracker export CSV into the portfolio.csv format
expected by agent.py. Built-in support for Finanzfluss; add a new ExportConfig
for other portfolio apps.

ISIN -> Yahoo Finance ticker resolution (in priority order):
  1. isin_map.json   -- personal overrides (gitignored, optional)
  2. OpenFIGI        -- Bloomberg free bulk API, 100 ISINs per HTTP request
  3. yfinance Search -- slow last resort; unreliable for non-US tickers

Set OPENFIGI_API_KEY in .env for higher rate limits (250 req/min vs 25).

Usage:
    python import_portfolio.py                        # Finanzfluss defaults
    python import_portfolio.py --input my.csv --output portfolio.csv
    python import_portfolio.py --quiet
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()


# -- Export format config ------------------------------------------------------

@dataclass
class ExportConfig:
    """Column mapping and number-parsing rules for one portfolio-app export format."""
    col_name: str
    col_isin: str
    col_shares: str
    col_price: str
    col_type: str
    excluded_types: set[str] = field(default_factory=set)
    parse_number: Callable[[str], float] = float


def parse_german_float(raw: str) -> float:
    """Convert a German-locale number string to float. '1.234,56' -> 1234.56"""
    s = raw.strip().strip('"').strip()
    if not s or s == "N/A":
        return 0.0
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


# Built-in config for Finanzfluss exports
FINANZFLUSS = ExportConfig(
    col_name="Name",
    col_isin="ISIN",
    col_shares="Anzahl",
    col_price="Kaufpreis",
    col_type="Typ",
    excluded_types={"Zertifikate/OS"},
    parse_number=parse_german_float,
)

# -- Output format -------------------------------------------------------------
OUTPUT_COLUMNS = ("ticker", "name", "shares", "avg_buy_price")

# -- ISIN -> Yahoo Finance ticker overrides ------------------------------------
# Loaded from isin_map.json (gitignored) at startup.
# Use this to correct cases where OpenFIGI picks the wrong exchange listing --
# most common with ETFs that trade on multiple exchanges.
# Copy isin_map.json.example to isin_map.json and add your own ISINs.
ISIN_TO_TICKER: dict[str, str] = {}
_isin_map_path = Path("isin_map.json")
if _isin_map_path.exists():
    ISIN_TO_TICKER.update(json.loads(_isin_map_path.read_text(encoding="utf-8-sig")))

# -- OpenFIGI -> Yahoo Finance -------------------------------------------------

# OpenFIGI exchange code -> Yahoo Finance ticker suffix
_EXCH_TO_SUFFIX: dict[str, str] = {
    "US": "", "UN": "", "UW": "", "UR": "", "UA": "",  # US exchanges, no suffix
    "GY": ".DE",   # Xetra (Deutsche Borse)
    "LN": ".L",    # London Stock Exchange
    "FP": ".PA",   # Euronext Paris
    "NA": ".AS",   # Euronext Amsterdam
    "SW": ".SW",   # SIX Swiss Exchange
    "AV": ".VI",   # Vienna
    "BB": ".BR",   # Brussels
    "IM": ".MI",   # Milan
    "SM": ".MC",   # Madrid
    "OL": ".OL",   # Oslo
    "DC": ".CO",   # Copenhagen
    "SS": ".ST",   # Stockholm
    "HK": ".HK",   # Hong Kong
    "AU": ".AX",   # ASX
    "JP": ".T",    # Tokyo
    "CT": ".TO",   # Toronto
}

# Preferred exchange codes by ISIN country prefix (first match wins)
_PREFIX_EXCH_PRIORITY: dict[str, list[str]] = {
    "US": ["US", "UN", "UW", "UR", "UA"],
    "CA": ["CT", "US", "UN"],
    "DE": ["GY"],
    "GB": ["LN"],
    "FR": ["FP"],
    "NL": ["NA"],
    "CH": ["SW"],
    "DK": ["DC"],
    "NO": ["OL"],
    "SE": ["SS"],
    "HK": ["HK"],
    "CN": ["HK"],
    "AU": ["AU"],
    "JP": ["JP"],
    "KY": ["US", "UN", "UW"],   # Cayman Islands -> US-listed
    "IE": ["GY", "LN"],         # UCITS ETFs -> Xetra first
    "LU": ["GY", "LN"],         # Luxembourg ETFs -> Xetra first
}

_OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
_OPENFIGI_BATCH_SIZE = 100


def _pick_ticker_from_figi(isin: str, candidates: list[dict]) -> str | None:
    """Select the best Yahoo Finance ticker from OpenFIGI candidate results."""
    priority = _PREFIX_EXCH_PRIORITY.get(isin[:2].upper(), [])
    for exch in priority:
        for c in candidates:
            if c.get("exchCode") == exch and c.get("ticker"):
                return c["ticker"] + _EXCH_TO_SUFFIX.get(exch, "")
    for c in candidates:
        exch = c.get("exchCode", "")
        ticker = c.get("ticker", "")
        if ticker and exch in _EXCH_TO_SUFFIX:
            return ticker + _EXCH_TO_SUFFIX[exch]
    return None


def bulk_openfigi_lookup(isins: list[str], verbose: bool = True) -> dict[str, str | None]:
    """Map a list of ISINs to Yahoo Finance tickers via OpenFIGI (100 per request)."""
    if not isins:
        return {}
    try:
        import requests
    except ImportError:
        if verbose:
            print("  [openfigi] 'requests' not installed -- skipping.", file=sys.stderr)
        return {isin: None for isin in isins}

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENFIGI_API_KEY")
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key

    out: dict[str, str | None] = {}
    for i in range(0, len(isins), _OPENFIGI_BATCH_SIZE):
        batch = isins[i : i + _OPENFIGI_BATCH_SIZE]
        payload = [{"idType": "ID_ISIN", "idValue": isin} for isin in batch]
        try:
            resp = requests.post(_OPENFIGI_URL, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            results = resp.json()
        except Exception as exc:
            if verbose:
                print(f"  [openfigi] Request failed: {exc}", file=sys.stderr)
            for isin in batch:
                out[isin] = None
            continue
        for isin, result in zip(batch, results):
            if "error" in result or not result.get("data"):
                out[isin] = None
            else:
                out[isin] = _pick_ticker_from_figi(isin, result["data"])
    return out


# -- yfinance last-resort fallback ---------------------------------------------

def try_yfinance_search(isin: str, name: str) -> str | None:
    """Resolve an unmapped ISIN via yfinance Search API (slow, unreliable)."""
    try:
        import yfinance as yf
        try:
            s = yf.Search(isin, max_results=1)
            quotes = getattr(s, "quotes", [])
            if quotes and quotes[0].get("symbol"):
                return quotes[0]["symbol"]
        except Exception:
            pass
        time.sleep(0.3)
        short_name = name.split("(")[0].strip()
        try:
            s = yf.Search(short_name, max_results=1)
            quotes = getattr(s, "quotes", [])
            if quotes and quotes[0].get("symbol"):
                return quotes[0]["symbol"]
        except Exception:
            pass
    except ImportError:
        pass
    return None


# -- ISIN resolution -----------------------------------------------------------

def resolve_ticker(
    isin: str,
    name: str,
    openfigi_cache: dict[str, str | None] | None = None,
    verbose: bool = True,
) -> str | None:
    """Resolve ISIN -> Yahoo Finance ticker. Priority: overrides -> OpenFIGI -> yfinance."""
    if isin in ISIN_TO_TICKER:
        return ISIN_TO_TICKER[isin]
    if openfigi_cache and openfigi_cache.get(isin):
        return openfigi_cache[isin]
    if verbose:
        print(f"  [search] OpenFIGI miss for {isin} ({name}) -- trying yfinance...")
    return try_yfinance_search(isin, name)


# -- CSV I/O -------------------------------------------------------------------

def read_export_csv(filepath: str) -> list[dict]:
    path = Path(filepath)
    if not path.exists():
        print(f"Error: '{filepath}' not found.")
        sys.exit(1)
    with open(filepath, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# -- Main conversion -----------------------------------------------------------

def convert(
    input_path: str = "investments.csv",
    output_path: str = "portfolio.csv",
    cfg: ExportConfig = FINANZFLUSS,
    verbose: bool = True,
) -> tuple[int, list[str]]:
    """
    Convert a portfolio-app export CSV to portfolio.csv.

    Returns (number_of_holdings_written, list_of_warning_strings).
    """
    rows = read_export_csv(input_path)
    if rows:
        required_cols = {cfg.col_name, cfg.col_isin, cfg.col_shares, cfg.col_price, cfg.col_type}
        missing_cols = required_cols - set(rows[0].keys())
        if missing_cols:
            print(f"Error: '{input_path}' is missing columns: {', '.join(sorted(missing_cols))}")
            sys.exit(1)
    if verbose:
        print(f"Read {len(rows)} rows from '{input_path}'.")

    # Filter: drop excluded types and zero-share positions
    active_rows = []
    for row in rows:
        if row.get(cfg.col_type, "").strip() in cfg.excluded_types:
            continue
        if cfg.parse_number(row.get(cfg.col_shares, "0")) <= 0:
            continue
        active_rows.append(row)

    # Collect ISINs not in the overrides map for bulk OpenFIGI lookup
    seen: set[str] = set()
    unmapped: list[str] = []
    for row in active_rows:
        isin = row.get(cfg.col_isin, "").strip()
        if isin not in ISIN_TO_TICKER and isin not in seen:
            unmapped.append(isin)
            seen.add(isin)

    openfigi_cache: dict[str, str | None] = {}
    if unmapped:
        if verbose:
            print(f"Looking up {len(unmapped)} ISIN(s) via OpenFIGI...")
        openfigi_cache = bulk_openfigi_lookup(unmapped, verbose=verbose)

    # Build holdings list
    holdings: list[dict] = []
    warnings: list[str] = []

    for row in active_rows:
        name = row.get(cfg.col_name, "").strip()
        isin = row.get(cfg.col_isin, "").strip()
        shares = cfg.parse_number(row.get(cfg.col_shares, "0"))
        price = cfg.parse_number(row.get(cfg.col_price, "0"))

        ticker = resolve_ticker(isin, name, openfigi_cache, verbose=verbose)

        if ticker is None:
            msg = (
                f"WARNING: Could not map ISIN '{isin}' ({name}) to a Yahoo Finance ticker. "
                f"Add it to isin_map.json or check the ISIN on finance.yahoo.com."
            )
            warnings.append(msg)
            if verbose:
                print(f"  {msg}")
            continue

        holdings.append({
            "ticker": ticker,
            "name": name,
            "shares": f"{shares:.6g}",
            "avg_buy_price": f"{price:.4f}",
        })

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OUTPUT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(holdings)

    if verbose:
        print(f"\nWrote {len(holdings)} active holdings to '{output_path}'.")
        if warnings:
            print(f"\n{len(warnings)} ISIN(s) could not be mapped -- add them to isin_map.json:")
            for w in warnings:
                print(f"  * {w}")

    return len(holdings), warnings


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a portfolio-app export CSV to portfolio.csv for agent.py"
    )
    parser.add_argument(
        "--input",
        default="investments.csv",
        help="Path to the export CSV (default: investments.csv)",
    )
    parser.add_argument(
        "--output",
        default="portfolio.csv",
        help="Path for the output portfolio.csv (default: portfolio.csv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-row output; only print summary and warnings",
    )
    args = parser.parse_args()

    _, warnings = convert(args.input, args.output, FINANZFLUSS, verbose=not args.quiet)
    if warnings:
        sys.exit(1)


if __name__ == "__main__":
    main()

