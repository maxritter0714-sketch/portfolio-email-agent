"""FX rates, prices, benchmarks and the figures derived from them."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from datetime import datetime as dt
from typing import Any

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# Return table in the email (1W/2W/1M/YTD): US-listed ETFs converted to EUR.
# They include dividends and close in New York like most holdings; a Xetra ETF
# closes at 17:35 CET and can sit a point or more off over a single week.
BENCHMARKS = [
    ("SPY", "S&P 500"),
    ("QQQ", "Nasdaq 100"),
    ("URTH", "MSCI World"),
]
# Risk figures in the email and everything in the dashboard: accumulating
# iShares ETFs on Xetra, in EUR, net of fund fees and withholding tax, i.e. what
# a German saver could hold instead. Beta, alpha and correlation use weekly
# returns, which the earlier close does not disturb.
BENCHMARK_ETFS: dict[str, tuple[str, str]] = {
    "sp500": ("SXR8.DE", "S&P 500"),
    "world": ("EUNL.DE", "MSCI World"),
}
# The S&P 500 price index (no dividends) only dates the history rows and feeds
# the history chart: stored index levels stay comparable over time, which
# dividend-adjusted ETF closes do not.
SP500_INDEX = "^GSPC"

# Closes fetched during this run, by symbol, so no series is downloaded twice.
_hist_cache: dict[str, pd.Series] = {}


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


# ── Market data ───────────────────────────────────────────────────────────────
_FORTNIGHT_BARS = 11  # trading days; week uses 6, month uses 22

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


# Currencies whose rate the last get_fx_rates() call could not fetch.
_fx_fallback_used: set[str] = set()


def fx_fallback_currencies() -> set[str]:
    """Currencies priced at a hardcoded constant in the last `get_fx_rates()` call."""
    return set(_fx_fallback_used)


def get_fx_rates() -> dict[str, float]:
    """EUR per unit of each currency; a failed lookup uses a constant and is remembered."""
    rates: dict[str, float] = {"EUR": 1.0}
    _fx_fallback_used.clear()
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
            _fx_fallback_used.add(currency)
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


def cached_closes(symbol: str) -> pd.Series | None:
    """Closes this run already fetched for `symbol`, or None."""
    return _hist_cache.get(symbol)


def clear_price_caches() -> None:
    """Forget every cached price and FX series, so the next lookup fetches again."""
    _hist_cache.clear()
    _fx_hist_cache.clear()


def _naive_dates(series: pd.Series) -> pd.Series:
    """Strip timezone and time-of-day so price and FX calendars can join."""
    out = series.copy()
    idx = out.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = pd.DatetimeIndex(idx).normalize()
    return out


def _fx_series(currency: str, period: str = "1y") -> pd.Series | None:
    """Daily EUR per 1 unit of `currency` over `period`. None means use a constant fallback."""
    ccy = "GBP" if currency in ("GBp", "GBX") else currency
    if ccy == "EUR":
        return None
    key = ccy if period == "1y" else f"{ccy}:{period}"
    if key in _fx_hist_cache:
        return _fx_hist_cache[key]
    pair = _FX_PAIRS.get(ccy)
    if not pair:
        log.warning("No FX pair for %s — constant fallback for EUR conversion", ccy)
        _fx_hist_cache[key] = None
        return None
    try:
        sym, invert = pair
        hist = yf.Ticker(sym).history(period=period)["Close"].dropna()
        rates = (1.0 / hist) if invert else hist
        rates = _naive_dates(rates)
        _fx_hist_cache[key] = rates
        return rates
    except Exception as exc:
        log.warning("Could not fetch FX history for %s (%s)", ccy, exc)
        _fx_hist_cache[key] = None
        return None


def _constant_eur_rate(currency: str, fx_rates: dict[str, float]) -> float:
    if currency in ("GBp", "GBX"):
        return fx_rates.get("GBp", _FX_FALLBACKS["GBp"])
    if currency in fx_rates:
        return fx_rates[currency]
    return _FX_FALLBACKS.get(currency, _FX_FALLBACKS["USD"])


def _to_eur_close(close: pd.Series, currency: str, fx_rates: dict[str, float],
                  fx_period: str = "1y") -> pd.Series:
    """Convert a native-currency close series to EUR using daily FX when available.

    `fx_period` must cover the price series, otherwise older prices are
    converted at the oldest available rate.
    """
    px = _naive_dates(close.astype(float))
    if currency in ("GBp", "GBX"):
        px = px / 100.0
        currency = "GBP"
    if currency == "EUR":
        return px
    fx = _fx_series(currency, fx_period)
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
    """Return since the last close of the prior year, as in the dashboard.

    A name without a close before `year` (listed this year) is measured from
    its first close.
    """
    vals = close.dropna()
    if vals.empty:
        return 0.0
    current = float(vals.iloc[-1])
    ytd = vals[vals.index.year == year]
    if ytd.empty:
        return 0.0
    prior = vals[vals.index.year < year]
    ref = float(prior.iloc[-1] if not prior.empty else ytd.iloc[0])
    return (current - ref) / ref * 100 if ref else 0.0


def _listing_currency(ticker: Any, symbol: str) -> str:
    """Currency the symbol is quoted in, from Yahoo's quote data or the history metadata.

    Raises when Yahoo reports none. Assuming USD would misprice a holding
    quoted in another currency without any visible sign of it.
    """
    for read in (lambda: ticker.fast_info.currency,
                 lambda: ticker.history_metadata.get("currency")):
        try:
            currency = read()
        except Exception:
            continue
        if currency:
            return currency
    raise ValueError(f"No listing currency returned for {symbol}")


def _bars_up_to(hist: pd.DataFrame, as_of: date | None) -> pd.DataFrame:
    """Drop price bars dated after `as_of`, by the exchange's own calendar.

    A run that starts after an Asian market has reopened would otherwise price
    that holding mid-session, under the date of the previous US close.
    """
    if as_of is None or hist.empty:
        return hist
    days = pd.DatetimeIndex(hist.index).tz_localize(None).normalize()
    return hist[days <= pd.Timestamp(as_of)]


@dataclass(frozen=True)
class PriceData:
    """One symbol's latest price, window returns and fund data."""

    price: float                # latest close, in the listing currency
    weekly_pct: float
    fortnight_pct: float
    monthly_pct: float
    ytd_pct: float
    currency: str               # listing currency as Yahoo reports it
    info: dict = field(default_factory=dict)
    sector_weights: dict[str, float] = field(default_factory=dict)    # ETFs only
    holdings_weights: dict[str, float] = field(default_factory=dict)  # ETFs only
    dividends_ttm: float = 0.0  # per share over the last year, in the listing currency


def get_price_changes(
    symbol: str,
    fx_rates: dict[str, float] | None = None,
    convert_to_eur: bool = False,
    as_of: date | None = None,
) -> PriceData:
    """Price, window returns and fund data for one symbol; `as_of` ignores later bars."""
    ticker = yf.Ticker(symbol)
    hist = _bars_up_to(ticker.history(period="1y"), as_of)
    if hist.empty:
        raise ValueError(f"No price history returned for {symbol}")

    # Drop trailing rows with no Close yet — e.g. the exchange's session is
    # still open or today's bar hasn't been finalized at fetch time — so
    # "current" never picks up a NaN price.
    hist = hist[hist["Close"].notna()]
    if hist.empty:
        raise ValueError(f"No finalized price history returned for {symbol}")

    currency = _listing_currency(ticker, symbol)

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

    dividends_ttm = float(hist["Dividends"].sum()) if "Dividends" in hist.columns else 0.0

    _hist_cache[symbol] = hist["Close"]
    return PriceData(
        price=current, weekly_pct=weekly_pct, fortnight_pct=fortnight_pct,
        monthly_pct=monthly_pct, ytd_pct=ytd_pct, currency=currency, info=info,
        sector_weights=sector_weights, holdings_weights=holdings_weights,
        dividends_ttm=dividends_ttm,
    )


def _expense_ratio(info: dict) -> float | None:
    """ETF ongoing charge as a fraction (Yahoo reports percent, 0.07 = 0.07 %); None for stocks."""
    ter = info.get("netExpenseRatio") if info.get("quoteType") == "ETF" else None
    return ter / 100 if isinstance(ter, (int, float)) else None


def portfolio_return(positions: list[dict], key: str) -> float:
    """Percent return of today's holdings held unchanged over the window behind `key`.

    Each position's start value is its value today divided by its growth, so
    the result is total value now over total value then. Averaging the
    position returns by today's weights instead would overweight the winners.
    """
    end = start = 0.0
    for p in positions:
        growth = 1.0 + p[key] / 100.0
        if growth <= 0:
            continue
        end += p["value_eur"]
        start += p["value_eur"] / growth
    return (end / start - 1.0) * 100.0 if start else 0.0


def fetch_portfolio_data(
    rows: list[dict], fx_rates: dict[str, float], as_of: date | None = None,
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    """Positions and portfolio summary; `as_of` values every holding at its close on or before that day."""
    positions = []
    unknown_fx: list[dict] = []
    skipped: list[dict] = []
    total_value_eur = 0.0
    total_cost_eur = 0.0

    for row in rows:
        symbol = row["ticker"].strip()
        name = row["name"].strip()
        shares = float(row["shares"])
        avg_buy_eur = float(row["avg_buy_price"])

        try:
            quote = get_price_changes(symbol, fx_rates=fx_rates, convert_to_eur=True, as_of=as_of)
        except Exception as exc:
            log.error("Skipping %s – could not fetch data: %s", symbol, exc)
            skipped.append({"ticker": symbol, "name": name, "reason": str(exc)})
            continue

        currency = quote.currency
        to_eur, used_usd_fallback = to_eur_rate(currency, fx_rates)
        if used_usd_fallback:
            unknown_fx.append({"ticker": symbol, "name": name, "currency": currency})
        price_eur = quote.price * to_eur
        value_eur = shares * price_eur
        cost_eur = shares * avg_buy_eur
        pl_eur = value_eur - cost_eur
        pl_pct = pl_eur / cost_eur * 100 if cost_eur else 0.0
        dividend_ttm_eur = shares * quote.dividends_ttm * to_eur

        sector, region = get_ticker_classification(symbol, quote.info)
        positions.append(
            dict(
                ticker=symbol, name=name, shares=shares,
                avg_buy_eur=avg_buy_eur, currency=currency,
                price_eur=price_eur, value_eur=value_eur, cost_eur=cost_eur,
                pl_eur=pl_eur, pl_pct=pl_pct,
                weekly_pct=quote.weekly_pct, fortnight_pct=quote.fortnight_pct,
                monthly_pct=quote.monthly_pct, ytd_pct=quote.ytd_pct,
                sector=sector, region=region,
                sector_weights=quote.sector_weights,
                holdings_weights=quote.holdings_weights,
                dividend_ttm_eur=dividend_ttm_eur,
                expense_ratio=_expense_ratio(quote.info),
            )
        )

        total_value_eur += value_eur
        total_cost_eur += cost_eur

    log.info("Fetched data for %d / %d positions", len(positions), len(rows))

    total_pl_eur = total_value_eur - total_cost_eur
    total_pl_pct = total_pl_eur / total_cost_eur * 100 if total_cost_eur else 0.0

    port_weekly = portfolio_return(positions, "weekly_pct")
    port_fortnight = portfolio_return(positions, "fortnight_pct")
    port_monthly = portfolio_return(positions, "monthly_pct")
    port_ytd = portfolio_return(positions, "ytd_pct")

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


def year_of_closes(symbol: str) -> pd.Series:
    """Last year of adjusted closes, from the price cache when the run already fetched them."""
    cached = _hist_cache.get(symbol)
    if cached is not None:
        return cached
    close = yf.Ticker(symbol).history(period="1y")["Close"]
    _hist_cache[symbol] = close
    return close


def sp500_index_close() -> pd.Series | None:
    """Recent closes of the S&P 500 price index (cached), or None if Yahoo has none."""
    cached = _hist_cache.get(SP500_INDEX)
    if cached is not None:
        return cached
    try:
        close = yf.Ticker(SP500_INDEX).history(period="5d")["Close"].dropna()
    except Exception as exc:
        log.warning("Could not fetch %s (%s)", SP500_INDEX, exc)
        return None
    if close.empty:
        return None
    _hist_cache[SP500_INDEX] = close
    return close


def fetch_benchmarks(fx_rates: dict[str, float]) -> list[dict]:
    """Benchmark returns in EUR, so they compare like-for-like with the portfolio."""
    results = []
    for symbol, label in BENCHMARKS:
        try:
            quote = get_price_changes(symbol, fx_rates=fx_rates, convert_to_eur=True)
            results.append(
                dict(symbol=symbol, name=label, weekly_pct=quote.weekly_pct,
                     fortnight_pct=quote.fortnight_pct, monthly_pct=quote.monthly_pct,
                     ytd_pct=quote.ytd_pct)
            )
            log.info("Benchmark %s: 1W %+.2f%% 1M %+.2f%% YTD %+.2f%%",
                     label, quote.weekly_pct, quote.monthly_pct, quote.ytd_pct)
        except Exception as exc:
            log.error("Failed to fetch benchmark %s: %s", symbol, exc)
    return results


# ── Recommendation track record prices ────────────────────────────────────────
def recommendation_prices(
    positions: list[dict], fx_rates: dict[str, float],
) -> tuple[pd.DataFrame, pd.Series]:
    """Last year of EUR closes for the track record: the holdings and the MSCI World ETF.

    Closes are split- and dividend-adjusted, so a split after a recommendation
    doesn't show up as a fake crash. Holdings come from the run's price cache
    (dates × tickers); the benchmark series is empty if Yahoo has no data.
    """
    cols: dict[str, pd.Series] = {}
    for p in positions:
        close = _hist_cache.get(p["ticker"])
        if close is None or close.empty:
            continue
        cols[p["ticker"]] = _to_eur_close(close, p.get("currency", "USD"), fx_rates).dropna()
    symbol = BENCHMARK_ETFS["world"][0]
    try:
        world = _naive_dates(year_of_closes(symbol).astype(float)).dropna()
    except Exception as exc:
        log.warning("Could not fetch %s for the track record (%s)", symbol, exc)
        world = pd.Series(dtype=float)
    return pd.DataFrame(cols).sort_index(), world


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
_UNASSIGNED = "Unassigned"


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


# ── Lifetime return ────────────────────────────────────────────────────────────
def lifetime_return(current_value: float, cashflows: list[dict]) -> dict | None:
    """Gain since the first cash-flow booking: value minus everything net paid in.

    Unlike the cost-basis P&L this includes realised gains and losses of sold
    positions. Assumes every purchase ran through the tracked cash account.
    None without cash-flow data or when nothing was paid in.
    """
    if not cashflows:
        return None
    invested = sum(c["invested_eur"] for c in cashflows)
    if invested <= 0:
        return None
    gain = current_value - invested
    return dict(
        since=date.fromisoformat(cashflows[0]["date"]),
        invested=invested,
        gain=gain,
        pct=gain / invested * 100,
        income=sum(c["income_eur"] for c in cashflows),
    )


