"""Market data, FX, history, and chart generation."""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import tempfile
from datetime import date, datetime as dt, timedelta
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from theme import (
    BG_CHART, BURGUNDY, DARK_GOLD, NAVY, SAGE,
    TEXT_PRIMARY, TEXT_SECONDARY, _PALETTE,
)

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"ticker", "name", "shares", "avg_buy_price"}
BENCHMARKS = [
    ("^GSPC", "S&P 500"),
    ("^IXIC", "Nasdaq"),
    ("URTH", "MSCI World"),
]

_hist_cache: dict = {}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif", "serif"],
    "figure.facecolor": BG_CHART,
    "axes.facecolor":   BG_CHART,
    "axes.edgecolor":   TEXT_PRIMARY,
    "axes.labelcolor":  TEXT_PRIMARY,
    "text.color":       TEXT_PRIMARY,
    "xtick.color":      TEXT_PRIMARY,
    "ytick.color":      TEXT_PRIMARY,
    "axes.titlesize":   12,
})

# ── Ticker classification ──────────────────────────────────────────────────────
# Normalises ETF sector_weightings keys (snake_case) to match stock sector
# labels returned by info["sector"] (Title Case).
_ETF_SECTOR_MAP: dict[str, str] = {
    "technology":             "Technology",
    "healthcare":             "Healthcare",
    "financial_services":     "Financial Services",
    "consumer_cyclical":      "Consumer Cyclical",
    "consumer_defensive":     "Consumer Defensive",
    "communication_services": "Communication Services",
    "industrials":            "Industrials",
    "energy":                 "Energy",
    "utilities":              "Utilities",
    "basic_materials":        "Basic Materials",
    "realestate":             "Real Estate",
}

# Keyword → region mapping for ETF longName classification.
# Checked in order — first match wins.
_ETF_REGION_KEYWORDS: list[tuple[str, str]] = [
    ("S&P 500",          "North America"),
    ("NASDAQ",           "North America"),
    ("Nasdaq",           "North America"),
    ("North America",    "North America"),
    ("Europe",           "Europe"),
    ("Swiss",            "Europe"),
    ("Japan",            "Japan"),
    ("Emerging Market",  "Emerging Markets"),
    ("China A",          "China / HK"),
    ("China",            "China / HK"),
    ("Asia",             "Asia Pacific"),
    ("World",            "Global"),
    ("Global",           "Global"),
]

_REGION_BUCKET: dict[str, str] = {
    "Germany": "Europe", "France": "Europe", "Switzerland": "Europe",
    "Italy": "Europe", "United Kingdom": "Europe", "UK": "Europe",
    "Netherlands": "Europe", "Denmark": "Europe", "Norway": "Europe",
    "Sweden": "Europe", "Spain": "Europe", "Belgium": "Europe",
    "Finland": "Europe", "Ireland": "Europe", "Austria": "Europe",
    "Portugal": "Europe", "Europe": "Europe",
    "United States": "North America", "Canada": "North America",
    "Japan": "Japan",
    "China": "China / HK", "Hong Kong": "China / HK",
    "Taiwan": "Asia Pacific", "South Korea": "Asia Pacific",
    "Singapore": "Asia Pacific", "Australia": "Asia Pacific",
    "India": "Emerging Markets", "Emerging Markets": "Emerging Markets",
    "Brazil": "Latin America", "Uruguay": "Latin America",
    "Mexico": "Latin America", "Argentina": "Latin America",
    "Israel": "Middle East",
    "Cayman Islands": "Other", "Marshall Islands": "Other",
    "Global": "Global",
}


def get_ticker_classification(symbol: str, info: dict) -> tuple[str, str]:
    """Return (sector, region) from yfinance info.

    ETF sector is always 'ETF' — actual sector distribution is handled via
    sector_weights in the position dict. ETF region is derived from longName
    via keyword matching so no manual overrides are needed.
    """
    quote_type = info.get("quoteType", "")
    if quote_type == "ETF":
        long_name = info.get("longName") or symbol
        region = "Global"
        for keyword, reg in _ETF_REGION_KEYWORDS:
            if keyword in long_name:
                region = reg
                break
        return "ETF", region
    sector = info.get("sector") or "Other"
    country = info.get("country") or "Other"
    return sector, _REGION_BUCKET.get(country, country)


# ── Cadence ───────────────────────────────────────────────────────────────────
# First send Saturday (issue № 1). Later sends are every 14 days from this date
# so the gap stays a true fortnight — ISO even-week would stretch to 21 days
# in years that have ISO week 53.
_SEND_EPOCH = date(2025, 1, 11)
_FORTNIGHT_BARS = 11  # trading days; week uses 6, month uses 22


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


# ── Portfolio loading ─────────────────────────────────────────────────────────
def load_portfolio(filepath: str = "portfolio.csv") -> list[dict]:
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


_EXPORT_PATH = Path("finanzfluss_export.csv")
_UNMAPPED_PATH = Path("unmapped_isins.json")


def holdings_as_of(portfolio_path: str = "portfolio.csv") -> date | None:
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


# ── Market data ───────────────────────────────────────────────────────────────
# Yahoo pair and whether to invert (EURXXX=X quotes XXX per 1 EUR).
_FX_PAIRS: dict[str, tuple[str, bool]] = {
    "USD": ("EURUSD=X", True),
    "GBP": ("GBPEUR=X", False),
    "HKD": ("EURHKD=X", True),
    "CNY": ("EURCNY=X", True),
    "CHF": ("EURCHF=X", True),
    "JPY": ("EURJPY=X", True),
    "DKK": ("EURDKK=X", True),
    "SEK": ("EURSEK=X", True),
    "NOK": ("EURNOK=X", True),
    "CAD": ("EURCAD=X", True),
    "AUD": ("EURAUD=X", True),
    "SGD": ("EURSGD=X", True),
    "KRW": ("EURKRW=X", True),
    "PLN": ("EURPLN=X", True),
    "NZD": ("EURNZD=X", True),
    "INR": ("EURINR=X", True),
    "TWD": ("EURTWD=X", True),
    "CZK": ("EURCZK=X", True),
    "HUF": ("EURHUF=X", True),
    "ILS": ("EURILS=X", True),
    "MXN": ("EURMXN=X", True),
    "BRL": ("EURBRL=X", True),
    "ZAR": ("EURZAR=X", True),
}
_FX_FALLBACKS: dict[str, float] = {
    "USD": 0.856, "GBP": 1.17, "GBp": 0.0117,
    "HKD": 0.110, "CNY": 0.138, "CHF": 1.12, "JPY": 0.006,
    "DKK": 0.134, "SEK": 0.092, "NOK": 0.086,
    "CAD": 0.63, "AUD": 0.57, "SGD": 0.67,
    "KRW": 0.00063, "PLN": 0.23, "NZD": 0.52,
    "INR": 0.010, "TWD": 0.028, "CZK": 0.041,
    "HUF": 0.0025, "ILS": 0.26, "MXN": 0.046,
    "BRL": 0.16, "ZAR": 0.049,
}


def get_fx_rates() -> dict[str, float]:
    rates: dict[str, float] = {"EUR": 1.0}
    for currency, (sym, invert) in _FX_PAIRS.items():
        try:
            hist = yf.Ticker(sym).history(period="1d")
            raw = float(hist["Close"].iloc[-1])
            rates[currency] = (1.0 / raw) if invert else raw
            log.info("FX %s: %.4f EUR", currency, rates[currency])
        except Exception as exc:
            fallback = _FX_FALLBACKS[currency]
            log.warning("Could not fetch %s rate (%s) — using fallback %.4f",
                        currency, exc, fallback)
            rates[currency] = fallback
    rates["GBp"] = rates.get("GBP", _FX_FALLBACKS["GBP"]) / 100
    rates["GBX"] = rates["GBp"]
    return rates


def to_eur_rate(currency: str, fx_rates: dict[str, float]) -> tuple[float, bool]:
    """Return (EUR per 1 unit of currency, used_usd_fallback)."""
    if currency in fx_rates:
        return fx_rates[currency], False
    usd = fx_rates.get("USD", _FX_FALLBACKS["USD"])
    log.warning("Unknown currency '%s' — using USD fallback %.4f", currency, usd)
    return usd, True


_fx_hist_cache: dict[str, pd.Series | None] = {}


def _naive_dates(series: pd.Series) -> pd.Series:
    """Strip timezone and time-of-day so price and FX calendars can join."""
    out = series.copy()
    idx = out.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = pd.DatetimeIndex(idx).normalize()
    return out


def _fx_series(currency: str) -> pd.Series | None:
    """1y daily EUR per 1 unit of `currency`. None means use a constant fallback."""
    ccy = "GBP" if currency in ("GBp", "GBX") else currency
    if ccy == "EUR":
        return None
    if ccy in _fx_hist_cache:
        return _fx_hist_cache[ccy]
    pair = _FX_PAIRS.get(ccy)
    if not pair:
        log.warning("No FX pair for %s — constant fallback for EUR conversion", ccy)
        _fx_hist_cache[ccy] = None
        return None
    try:
        sym, invert = pair
        hist = yf.Ticker(sym).history(period="1y")["Close"].dropna()
        rates = (1.0 / hist) if invert else hist
        rates = _naive_dates(rates)
        _fx_hist_cache[ccy] = rates
        return rates
    except Exception as exc:
        log.warning("Could not fetch FX history for %s (%s)", ccy, exc)
        _fx_hist_cache[ccy] = None
        return None


def _constant_eur_rate(currency: str, fx_rates: dict[str, float]) -> float:
    if currency in ("GBp", "GBX"):
        return fx_rates.get("GBp", _FX_FALLBACKS["GBp"])
    if currency in fx_rates:
        return fx_rates[currency]
    return _FX_FALLBACKS.get(currency, _FX_FALLBACKS["USD"])


def _to_eur_close(close: pd.Series, currency: str, fx_rates: dict[str, float]) -> pd.Series:
    """Convert a native-currency close series to EUR using daily FX when available."""
    px = _naive_dates(close.astype(float))
    if currency in ("GBp", "GBX"):
        px = px / 100.0
        currency = "GBP"
    if currency == "EUR":
        return px
    fx = _fx_series(currency)
    fallback = _constant_eur_rate(currency, fx_rates)
    if fx is None:
        return px * fallback
    aligned = fx.reindex(px.index).ffill().bfill()
    if aligned.isna().all():
        return px * fallback
    return px * aligned


def _window_pct(close: pd.Series, bars: int) -> float:
    vals = close.dropna()
    if vals.empty:
        return 0.0
    current = float(vals.iloc[-1])
    ref = float(vals.iloc[-bars] if len(vals) >= bars else vals.iloc[0])
    return (current - ref) / ref * 100 if ref else 0.0


def _ytd_pct_from_close(close: pd.Series, year: int) -> float:
    vals = close.dropna()
    if vals.empty:
        return 0.0
    current = float(vals.iloc[-1])
    ytd = vals[vals.index.year == year]
    if ytd.empty:
        return 0.0
    ref = float(ytd.iloc[0])
    return (current - ref) / ref * 100 if ref else 0.0


def get_price_changes(
    symbol: str,
    fx_rates: dict[str, float] | None = None,
    convert_to_eur: bool = False,
) -> tuple[float, float, float, float, float, str, dict, dict, dict, float]:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y")
    if hist.empty:
        raise ValueError(f"No price history returned for {symbol}")

    # Drop trailing rows with no Close yet — e.g. the exchange's session is
    # still open or today's bar hasn't been finalized at fetch time — so
    # "current" never picks up a NaN price.
    hist = hist[hist["Close"].notna()]
    if hist.empty:
        raise ValueError(f"No finalized price history returned for {symbol}")

    try:
        currency = ticker.fast_info.currency or "USD"
    except Exception:
        currency = "USD"

    current = float(hist["Close"].iloc[-1])
    close_for_pct = hist["Close"]
    if convert_to_eur:
        rates = fx_rates if fx_rates is not None else {"EUR": 1.0}
        close_for_pct = _to_eur_close(hist["Close"], currency, rates)

    weekly_pct = _window_pct(close_for_pct, 6)
    fortnight_pct = _window_pct(close_for_pct, _FORTNIGHT_BARS)
    monthly_pct = _window_pct(close_for_pct, 22)
    ytd_pct = _ytd_pct_from_close(close_for_pct, date.today().year)

    try:
        info = ticker.info
    except Exception:
        info = {}

    sector_weights: dict[str, float] = {}
    holdings_weights: dict[str, float] = {}
    if info.get("quoteType") == "ETF":
        try:
            raw = ticker.funds_data.sector_weightings
            sector_weights = {
                _ETF_SECTOR_MAP[k]: v
                for k, v in raw.items()
                if k in _ETF_SECTOR_MAP and v > 0
            }
        except Exception:
            pass
        try:
            top = ticker.funds_data.top_holdings
            if top is not None and not top.empty:
                holdings_weights = {}
                for key, value in top["Holding Percent"].to_dict().items():
                    try:
                        holdings_weights[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
        except Exception:
            pass

    div_ttm_native = float(hist["Dividends"].sum()) if "Dividends" in hist.columns else 0.0

    _hist_cache[symbol] = hist["Close"]
    return current, weekly_pct, fortnight_pct, monthly_pct, ytd_pct, currency, info, sector_weights, holdings_weights, div_ttm_native


def fetch_portfolio_data(
    rows: list[dict], fx_rates: dict[str, float]
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    positions = []
    unknown_fx: list[dict] = []
    skipped: list[dict] = []
    total_value_eur = 0.0
    total_cost_eur = 0.0
    weighted_weekly = weighted_fortnight = weighted_monthly = weighted_ytd = 0.0

    for row in rows:
        symbol = row["ticker"].strip()
        name = row["name"].strip()
        shares = float(row["shares"])
        avg_buy_eur = float(row["avg_buy_price"])

        try:
            price_native, weekly, fortnight, monthly, ytd, currency, info, sector_weights, holdings_weights, div_ttm_native = get_price_changes(
                symbol, fx_rates=fx_rates, convert_to_eur=True,
            )
        except Exception as exc:
            log.error("Skipping %s – could not fetch data: %s", symbol, exc)
            skipped.append({"ticker": symbol, "name": name, "reason": str(exc)})
            continue

        to_eur, used_usd_fallback = to_eur_rate(currency, fx_rates)
        if used_usd_fallback:
            unknown_fx.append({"ticker": symbol, "name": name, "currency": currency})
        price_eur = price_native * to_eur
        value_eur = shares * price_eur
        cost_eur = shares * avg_buy_eur
        pl_eur = value_eur - cost_eur
        pl_pct = pl_eur / cost_eur * 100 if cost_eur else 0.0
        dividend_ttm_eur = shares * div_ttm_native * to_eur

        sector, region = get_ticker_classification(symbol, info)
        positions.append(
            dict(
                ticker=symbol, name=name, shares=shares,
                avg_buy_eur=avg_buy_eur, currency=currency,
                price_eur=price_eur, value_eur=value_eur, cost_eur=cost_eur,
                pl_eur=pl_eur, pl_pct=pl_pct,
                weekly_pct=weekly, fortnight_pct=fortnight, monthly_pct=monthly, ytd_pct=ytd,
                sector=sector, region=region,
                sector_weights=sector_weights,
                holdings_weights=holdings_weights,
                dividend_ttm_eur=dividend_ttm_eur,
            )
        )

        total_value_eur += value_eur
        total_cost_eur += cost_eur
        weighted_weekly += weekly * value_eur
        weighted_fortnight += fortnight * value_eur
        weighted_monthly += monthly * value_eur
        weighted_ytd += ytd * value_eur

    log.info("Fetched data for %d / %d positions", len(positions), len(rows))

    total_pl_eur = total_value_eur - total_cost_eur
    total_pl_pct = total_pl_eur / total_cost_eur * 100 if total_cost_eur else 0.0

    port_weekly = weighted_weekly / total_value_eur if total_value_eur else 0.0
    port_fortnight = weighted_fortnight / total_value_eur if total_value_eur else 0.0
    port_monthly = weighted_monthly / total_value_eur if total_value_eur else 0.0
    port_ytd = weighted_ytd / total_value_eur if total_value_eur else 0.0

    total_dividends_ttm_eur = sum(p["dividend_ttm_eur"] for p in positions)
    portfolio_yield_pct = total_dividends_ttm_eur / total_value_eur * 100 if total_value_eur else 0.0

    summary = dict(
        total_value_eur=total_value_eur, total_cost_eur=total_cost_eur,
        total_pl_eur=total_pl_eur, total_pl_pct=total_pl_pct,
        weekly_pct=port_weekly, fortnight_pct=port_fortnight, monthly_pct=port_monthly, ytd_pct=port_ytd,
        total_dividends_ttm_eur=total_dividends_ttm_eur,
        portfolio_yield_pct=portfolio_yield_pct,
    )
    if skipped:
        log.warning(
            "Omitted %d ticker(s) with no price data: %s",
            len(skipped), ", ".join(s["ticker"] for s in skipped),
        )
    return positions, summary, unknown_fx, skipped


def fetch_benchmarks(fx_rates: dict[str, float]) -> list[dict]:
    """Benchmark returns in EUR, so they compare like-for-like with the portfolio."""
    results = []
    for symbol, label in BENCHMARKS:
        try:
            _, weekly, fortnight, monthly, ytd, _cur, _info, _sw, _hw, _div = get_price_changes(
                symbol, fx_rates=fx_rates, convert_to_eur=True,
            )
            results.append(
                dict(symbol=symbol, name=label, weekly_pct=weekly,
                     fortnight_pct=fortnight, monthly_pct=monthly, ytd_pct=ytd)
            )
            log.info("Benchmark %s: 1W %+.2f%% 1M %+.2f%% YTD %+.2f%%",
                     label, weekly, monthly, ytd)
        except Exception as exc:
            log.error("Failed to fetch benchmark %s: %s", symbol, exc)
    return results


# ── Chart generation ─────────────────────────────────────────────────────────
def _save_fig(fig, name: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".png", prefix=f"portfolio_{name}_")
    os.close(fd)
    fig.savefig(path, format="png", dpi=140, bbox_inches="tight",
                facecolor=BG_CHART, edgecolor="none")
    plt.close(fig)
    return path


def _prepare_pie(items: list[tuple[str, float]],
                 min_pct: float = 0.0,
                 max_slices: int | None = None) -> tuple[list[str], list[float]]:
    """Aggregate small items into 'Others'. Returns (labels, sizes)."""
    total = sum(v for _, v in items) or 1.0
    items_sorted = sorted(items, key=lambda x: -x[1])
    if min_pct > 0:
        cutoff = total * min_pct / 100.0
        large = [x for x in items_sorted if x[1] >= cutoff]
        small = [x for x in items_sorted if x[1] < cutoff]
    else:
        n = max_slices or len(items_sorted)
        large = items_sorted[:n]
        small = items_sorted[n:]
    labels = [l for l, _ in large]
    sizes = [v for _, v in large]
    if small:
        labels.append("Others")
        sizes.append(sum(v for _, v in small))
    return labels, sizes


def _pie(ax, sizes: list[float], labels: list[str]) -> None:
    wedges, _, autotexts = ax.pie(
        sizes,
        colors=_PALETTE[:len(sizes)],
        autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
        pctdistance=0.78,
        startangle=90,
        wedgeprops={"linewidth": 1.5, "edgecolor": BG_CHART},
        textprops={"fontsize": 8, "color": TEXT_PRIMARY, "fontfamily": "serif"},
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color(TEXT_PRIMARY)
    ax.legend(
        wedges, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=7.5,
        frameon=False,
        labelcolor=TEXT_PRIMARY,
    )


_HISTORY_PATH = Path("history.csv")
_HISTORY_COLS = (
    "date", "issue", "total_value_eur", "sp500_close", "datetime", "source",
    "port_index", "sp500_eur",
)
_HISTORY_CHART_MIN_ROWS = 4


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
        close = _hist_cache.get(p["ticker"])
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
) -> None:
    """Append one row to history.csv per (date, source) pair."""
    today_str = date.today().isoformat()
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
            return

    sp500_series = _hist_cache.get("^GSPC")
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


def generate_history_chart() -> str | None:
    """Line chart: portfolio price index vs S&P 500 in EUR, both rebased to 100.

    Uses the deposit-free port_index column, so cash in/out never shows up as
    performance. Rows written before that column existed are skipped.
    """
    # Keep only valid scheduled rows — manual runs don't belong in the trend chart.
    rows = [
        r for r in _read_history()
        if r.get("date")
        and _is_float(r.get("port_index"))
        and _is_float(r.get("sp500_eur"))
        and r.get("source", "scheduled") == "scheduled"
    ]
    if len(rows) < _HISTORY_CHART_MIN_ROWS:
        return None

    labels   = [r["date"] for r in rows]
    port_raw = [float(r["port_index"]) for r in rows]
    sp_raw   = [float(r["sp500_eur"]) for r in rows]

    base_port = port_raw[0] or 1.0
    base_sp   = sp_raw[0]   or 1.0
    port_idx  = [v / base_port * 100 for v in port_raw]
    sp_idx    = [v / base_sp   * 100 for v in sp_raw]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(labels)), port_idx, color=DARK_GOLD, linewidth=2,
            label="Portfolio", marker="o", markersize=4)
    ax.plot(range(len(labels)), sp_idx, color=NAVY, linewidth=1.5,
            linestyle="--", label="S&P 500 (EUR)", marker="o", markersize=3)
    ax.axhline(100, color=TEXT_SECONDARY, linewidth=0.5, linestyle=":")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [pd.Timestamp(r["date"]).strftime("%d %b '%y").lstrip("0") for r in rows],
        fontsize=7, color=TEXT_PRIMARY, fontfamily="serif", rotation=45, ha="right",
    )
    ax.tick_params(axis="y", labelsize=8, colors=TEXT_PRIMARY)
    ax.set_title("Price return in EUR, deposits excluded, indexed to 100 at start", fontsize=8,
                 color=TEXT_SECONDARY, fontfamily="serif", pad=6)
    ax.set_ylabel("Indexed (start = 100)", fontsize=8, color=TEXT_SECONDARY,
                  fontfamily="serif")
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT_PRIMARY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(TEXT_SECONDARY)
    ax.set_facecolor(BG_CHART)
    fig.patch.set_facecolor(BG_CHART)
    return _save_fig(fig, "history_line")


def generate_charts(positions: list[dict], summary: dict,
                    benchmarks: list[dict]) -> dict[str, str]:
    charts: dict[str, str] = {}

    history_path = generate_history_chart()
    if history_path:
        charts["history_line"] = history_path
    total = summary["total_value_eur"]

    # 1. Portfolio allocation — group positions <2% into Others
    items = [(p["ticker"], p["value_eur"]) for p in positions]
    labels, sizes = _prepare_pie(items, min_pct=2.0)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    _pie(ax, sizes, labels)
    ax.text(0, 0, f"€{total:,.0f}", ha="center", va="center",
            fontsize=12, color=TEXT_PRIMARY, fontfamily="serif")
    charts["portfolio_pie"] = _save_fig(fig, "portfolio_pie")

    # 2. Sector allocation — top 8 + Others
    # ETFs are distributed across their actual underlying sectors via sector_weights.
    sec: dict[str, float] = {}
    for p in positions:
        weights = p.get("sector_weights")
        if weights:
            for sector_name, weight in weights.items():
                sec[sector_name] = sec.get(sector_name, 0) + p["value_eur"] * weight
        else:
            s = p.get("sector") or "Other"
            sec[s] = sec.get(s, 0) + p["value_eur"]
    labels, sizes = _prepare_pie(list(sec.items()), max_slices=8)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    _pie(ax, sizes, labels)
    charts["sector_pie"] = _save_fig(fig, "sector_pie")

    # 3. Geographic allocation — top 8 + Others
    geo: dict[str, float] = {}
    for p in positions:
        region = p.get("region") or "Other"
        geo[region] = geo.get(region, 0) + p["value_eur"]
    labels, sizes = _prepare_pie(list(geo.items()), max_slices=8)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    _pie(ax, sizes, labels)
    charts["geo_pie"] = _save_fig(fig, "geo_pie")

    # 4. Top & Bottom movers (YTD)
    by_ytd = sorted(positions, key=lambda p: p["ytd_pct"])
    bottom5 = by_ytd[:5]
    top5 = list(reversed(by_ytd[-5:]))
    movers = top5 + bottom5
    names = [p["ticker"] for p in movers]
    vals = [p["ytd_pct"] for p in movers]
    colors = [SAGE if v >= 0 else BURGUNDY for v in vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(list(reversed(names)), list(reversed(vals)),
                   color=list(reversed(colors)), edgecolor=BG_CHART, linewidth=0.6)
    ax.axvline(0, color=TEXT_PRIMARY, linewidth=0.6)
    for bar, val in zip(bars, reversed(vals)):
        pad = max(abs(val) * 0.02, 0.3)
        ax.text(val + (pad if val >= 0 else -pad),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9, color=TEXT_PRIMARY, fontfamily="serif")
    ax.set_xlabel("YTD Return (%)", fontsize=10, fontfamily="serif")
    ax.tick_params(labelsize=10)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_PRIMARY)
    ax.spines["bottom"].set_linewidth(0.6)
    fig.tight_layout()
    charts["movers_bar"] = _save_fig(fig, "movers_bar")

    log.info("Generated %d charts", len(charts))
    return charts


# ── Recommendation reference prices ───────────────────────────────────────────
def recommendation_ref_prices(
    conviction: dict, positions: list[dict], fx_rates: dict[str, float],
) -> dict[str, float]:
    """Split-adjusted EUR close on each recommendation's date, from cached history.

    The price stored in conviction.json is unadjusted, so a stock split after
    the recommendation would show up as a fake crash. Tickers without history
    covering that date are left out; callers fall back to the stored price.
    """
    currency_of = {p["ticker"]: p.get("currency", "USD") for p in positions}
    out: dict[str, float] = {}
    for ticker, rec in conviction.items():
        close = _hist_cache.get(ticker)
        if ticker not in currency_of or close is None or close.empty or not rec.get("date"):
            continue
        eur = _to_eur_close(close, currency_of[ticker], fx_rates).dropna()
        before = eur[eur.index <= pd.Timestamp(rec["date"])]
        if not before.empty:
            out[ticker] = float(before.iloc[-1])
    return out


# ── Upcoming events ───────────────────────────────────────────────────────────
def _as_dates(value: object) -> list[date]:
    """yfinance calendar values are a date, a list of dates, or missing."""
    items = value if isinstance(value, (list, tuple)) else [value]
    out: list[date] = []
    for item in items:
        if isinstance(item, dt):
            out.append(item.date())
        elif isinstance(item, date):
            out.append(item)
    return out


def events_from_calendar(
    calendar: dict, ticker: str, name: str, today: date, days: int,
) -> list[dict]:
    """Earnings and ex-dividend dates from one yfinance calendar, within the window."""
    window_end = today + timedelta(days=days)
    events: list[dict] = []
    for key, kind in (("Earnings Date", "Earnings"), ("Ex-Dividend Date", "Ex-dividend")):
        for d in _as_dates(calendar.get(key)):
            if today <= d <= window_end:
                events.append(dict(date=d, ticker=ticker, name=name, kind=kind))
    return events


def fetch_upcoming_events(positions: list[dict], days: int = 14) -> list[dict]:
    """Earnings and ex-dividend dates for directly held stocks in the next `days`.

    ETFs are skipped — Yahoo publishes no calendar for them.
    """
    today = date.today()
    events: list[dict] = []
    for p in positions:
        if p.get("sector") == "ETF":
            continue
        try:
            calendar = yf.Ticker(p["ticker"]).calendar or {}
        except Exception as exc:
            log.debug("No calendar for %s: %s", p["ticker"], exc)
            continue
        events.extend(events_from_calendar(calendar, p["ticker"], p["name"], today, days))
    events.sort(key=lambda e: (e["date"], e["ticker"]))
    log.info("Upcoming events (next %d days): %d", days, len(events))
    return events


# ── Target allocation ─────────────────────────────────────────────────────────
_TARGETS_PATH = Path("targets.json")
_UNASSIGNED = "Unassigned"


def load_targets(path: Path = _TARGETS_PATH) -> dict | None:
    """Optional target allocation from targets.json, or None when not configured."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        log.warning("Could not read %s (%s) — skipping target allocation", path, exc)
        return None


def compute_allocation_drift(positions: list[dict], targets: dict) -> list[dict]:
    """Actual vs target weight per group. Tickers in no group land in 'Unassigned'."""
    total = sum(p["value_eur"] for p in positions)
    if not total:
        return []
    tolerance = float(targets.get("tolerance_pct", 5.0))
    group_of: dict[str, str] = {}
    rows: list[dict] = []
    for g in targets.get("groups", []):
        rows.append(dict(name=g["name"], target_pct=float(g["target_pct"]), value_eur=0.0))
        for t in g.get("tickers", []):
            group_of[t] = g["name"]
    by_name = {r["name"]: r for r in rows}

    unassigned = 0.0
    for p in positions:
        group = group_of.get(p["ticker"])
        if group is None:
            unassigned += p["value_eur"]
        else:
            by_name[group]["value_eur"] += p["value_eur"]
    if unassigned:
        rows.append(dict(name=_UNASSIGNED, target_pct=None, value_eur=unassigned))

    for r in rows:
        r["actual_pct"] = r["value_eur"] / total * 100
        if r["target_pct"] is None:
            r["drift_pct"] = None
            r["off_target"] = False
        else:
            r["drift_pct"] = r["actual_pct"] - r["target_pct"]
            r["off_target"] = abs(r["drift_pct"]) > tolerance
    return rows


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
