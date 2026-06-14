#!/usr/bin/env python3
"""
Portfolio Email Agent
Fetches market data, news, and Claude AI analysis, then sends an editorial-style
HTML portfolio report via Gmail. Runs automatically on the second Saturday of
each month; bypass with --force for testing.
"""

import argparse
import csv
import logging
import os
import re
import smtplib
import sys
import tempfile
import traceback
from datetime import date
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import anthropic
import yfinance as yf
from dotenv import load_dotenv
from newsapi import NewsApiClient

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = {"ticker", "name", "shares", "avg_buy_price"}
BENCHMARKS = [
    ("^GSPC", "S&P 500"),
    ("^IXIC", "Nasdaq"),
    ("URTH", "MSCI World"),
]
CLAUDE_MODEL = "claude-sonnet-4-6"

# ── Editorial colour palette ──────────────────────────────────────────────────
BG_CREAM        = "#fafaf7"
BG_CHART        = "#f1eee5"
CHARCOAL        = "#0f1419"
GOLD            = "#c9a961"
DARK_GOLD       = "#7a5f2a"
SAGE            = "#4a7d54"
BURGUNDY        = "#8b3a3a"
NAVY            = "#3a5a8b"
TEXT_PRIMARY    = "#1a1a1a"
TEXT_SECONDARY  = "#8a7b5e"
RULE_FAINT      = "rgba(26,26,26,0.1)"

# Chart palette — no default matplotlib blues/greens
_PALETTE = [
    GOLD, NAVY, SAGE, DARK_GOLD, BURGUNDY,
    "#dcc08a", "#6d87b0", "#7ba384", "#b06060", TEXT_SECONDARY,
    "#5d2525", "#253d5e", "#2f5438", "#9d8845", "#a09584", "#5a4a33",
]

if CHARTS_AVAILABLE:
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


# ── Date helpers ──────────────────────────────────────────────────────────────
def is_second_saturday() -> bool:
    today = date.today()
    if today.weekday() != 5:
        return False
    first_day = today.replace(day=1)
    days_to_first_sat = (5 - first_day.weekday()) % 7
    second_sat_day = 1 + days_to_first_sat + 7
    return today.day == second_sat_day


def issue_number(today: date | None = None) -> int:
    """Monthly issue count, with Jan 2025 as № 1."""
    today = today or date.today()
    return max(1, (today.year - 2025) * 12 + today.month)


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


# ── Market data ───────────────────────────────────────────────────────────────
def get_fx_rates() -> dict[str, float]:
    pairs = {
        "USD": ("EURUSD=X", True),
        "GBP": ("GBPEUR=X", False),
        "HKD": ("EURHKD=X", True),
        "CNY": ("EURCNY=X", True),
        "CHF": ("EURCHF=X", True),
        "JPY": ("EURJPY=X", True),
    }
    fallbacks = {
        "USD": 0.856, "GBP": 1.17, "GBp": 0.0117,
        "HKD": 0.110, "CNY": 0.138, "CHF": 1.12, "JPY": 0.006,
    }
    rates: dict[str, float] = {"EUR": 1.0}
    for currency, (sym, invert) in pairs.items():
        try:
            hist = yf.Ticker(sym).history(period="1d")
            raw = float(hist["Close"].iloc[-1])
            rates[currency] = (1.0 / raw) if invert else raw
            log.info("FX %s: %.4f EUR", currency, rates[currency])
        except Exception as exc:
            log.warning("Could not fetch %s rate (%s) — using fallback %.4f",
                        currency, exc, fallbacks[currency])
            rates[currency] = fallbacks[currency]
    rates["GBp"] = rates.get("GBP", fallbacks["GBP"]) / 100
    return rates


def get_price_changes(symbol: str) -> tuple[float, float, float, float, str]:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y")
    if hist.empty:
        raise ValueError(f"No price history returned for {symbol}")

    try:
        currency = ticker.fast_info.currency or "USD"
    except Exception:
        currency = "USD"

    current = float(hist["Close"].iloc[-1])

    week_ref = float(hist["Close"].iloc[-6] if len(hist) >= 6 else hist["Close"].iloc[0])
    weekly_pct = (current - week_ref) / week_ref * 100

    month_ref = float(hist["Close"].iloc[-22] if len(hist) >= 22 else hist["Close"].iloc[0])
    monthly_pct = (current - month_ref) / month_ref * 100

    this_year = date.today().year
    ytd_hist = hist[hist.index.year == this_year]
    if not ytd_hist.empty:
        ytd_ref = float(ytd_hist["Close"].iloc[0])
        ytd_pct = (current - ytd_ref) / ytd_ref * 100
    else:
        ytd_pct = 0.0

    try:
        info = ticker.info
    except Exception:
        info = {}

    sector_weights: dict[str, float] = {}
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

    return current, weekly_pct, monthly_pct, ytd_pct, currency, info, sector_weights


def fetch_portfolio_data(rows: list[dict], fx_rates: dict[str, float]) -> tuple[list[dict], dict]:
    positions = []
    total_value_eur = 0.0
    total_cost_eur = 0.0
    weighted_weekly = weighted_monthly = weighted_ytd = 0.0

    for row in rows:
        symbol = row["ticker"].strip()
        name = row["name"].strip()
        shares = float(row["shares"])
        avg_buy_eur = float(row["avg_buy_price"])

        try:
            price_native, weekly, monthly, ytd, currency, info, sector_weights = get_price_changes(symbol)
        except Exception as exc:
            log.error("Skipping %s – could not fetch data: %s", symbol, exc)
            continue

        to_eur = fx_rates.get(currency, fx_rates.get("USD", 0.856))
        price_eur = price_native * to_eur
        value_eur = shares * price_eur
        cost_eur = shares * avg_buy_eur
        pl_eur = value_eur - cost_eur
        pl_pct = pl_eur / cost_eur * 100 if cost_eur else 0.0

        sector, region = get_ticker_classification(symbol, info)
        positions.append(
            dict(
                ticker=symbol, name=name, shares=shares,
                avg_buy_eur=avg_buy_eur, currency=currency,
                price_eur=price_eur, value_eur=value_eur, cost_eur=cost_eur,
                pl_eur=pl_eur, pl_pct=pl_pct,
                weekly_pct=weekly, monthly_pct=monthly, ytd_pct=ytd,
                sector=sector, region=region,
                sector_weights=sector_weights,
            )
        )

        total_value_eur += value_eur
        total_cost_eur += cost_eur
        weighted_weekly += weekly * value_eur
        weighted_monthly += monthly * value_eur
        weighted_ytd += ytd * value_eur

    log.info("Fetched data for %d / %d positions", len(positions), len(rows))

    total_pl_eur = total_value_eur - total_cost_eur
    total_pl_pct = total_pl_eur / total_cost_eur * 100 if total_cost_eur else 0.0

    port_weekly = weighted_weekly / total_value_eur if total_value_eur else 0.0
    port_monthly = weighted_monthly / total_value_eur if total_value_eur else 0.0
    port_ytd = weighted_ytd / total_value_eur if total_value_eur else 0.0

    summary = dict(
        total_value_eur=total_value_eur, total_cost_eur=total_cost_eur,
        total_pl_eur=total_pl_eur, total_pl_pct=total_pl_pct,
        weekly_pct=port_weekly, monthly_pct=port_monthly, ytd_pct=port_ytd,
    )
    return positions, summary


def fetch_benchmarks() -> list[dict]:
    results = []
    for symbol, label in BENCHMARKS:
        try:
            _, weekly, monthly, ytd, _cur, _info, _sw = get_price_changes(symbol)
            results.append(
                dict(symbol=symbol, name=label, weekly_pct=weekly,
                     monthly_pct=monthly, ytd_pct=ytd)
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


def generate_charts(positions: list[dict], summary: dict,
                    benchmarks: list[dict]) -> dict[str, str]:
    if not CHARTS_AVAILABLE:
        log.warning("matplotlib not available — charts skipped")
        return {}

    charts: dict[str, str] = {}
    total = summary["total_value_eur"]

    # 1. Portfolio allocation — group positions <2% into Others
    items = [(p["ticker"], p["value_eur"]) for p in positions]
    labels, sizes = _prepare_pie(items, min_pct=2.0)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    _pie(ax, sizes, labels)
    ax.text(0, 0, f"€{total:,.0f}", ha="center", va="center",
            fontsize=12, color=DARK_GOLD, fontfamily="serif")
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
        raw = p.get("region") or "Other"
        bucket = _REGION_BUCKET.get(raw, raw)
        geo[bucket] = geo.get(bucket, 0) + p["value_eur"]
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


# ── Advanced metrics ──────────────────────────────────────────────────────────
def compute_advanced_metrics(positions: list[dict], fx_rates: dict[str, float]) -> dict:
    if not CHARTS_AVAILABLE:
        return {}

    RISK_FREE = 0.035
    TDAYS = 252

    log.info("Computing advanced metrics…")

    import pandas as pd

    total_val = sum(p["value_eur"] for p in positions)
    if total_val == 0:
        return {}

    ret_df: dict[str, object] = {}
    for p in positions:
        weight = p["value_eur"] / total_val
        try:
            hist = yf.Ticker(p["ticker"]).history(period="1y")["Close"]
            if hist.empty or len(hist) < 60:
                continue
            ret_df[p["ticker"]] = hist.pct_change().dropna() * weight
        except Exception:
            continue

    if not ret_df:
        log.warning("Not enough history for metrics")
        return {}

    df = pd.DataFrame(ret_df)
    port_ret_series = df.sum(axis=1, min_count=1).dropna()

    if len(port_ret_series) < 60:
        log.warning("Not enough history for metrics")
        return {}

    try:
        sp500_hist = yf.Ticker("^GSPC").history(period="1y")["Close"]
        sp500_ret = sp500_hist.pct_change().dropna()
    except Exception as exc:
        log.error("Failed to fetch S&P 500 for metrics: %s", exc)
        return {}

    common_idx = port_ret_series.index.intersection(sp500_ret.index)
    if len(common_idx) < 60:
        return {}

    pr = port_ret_series.loc[common_idx].values.astype(float)
    sr = sp500_ret.loc[common_idx].values.astype(float)

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
    correlation = float(np.corrcoef(pr, sr)[0, 1])

    total_val = sum(p["value_eur"] for p in positions)
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
    )


# ── News ──────────────────────────────────────────────────────────────────────
_NAME_STOP = {
    "inc", "inc.", "corp", "corp.", "corporation", "ltd", "ltd.", "plc",
    "group", "holdings", "holding", "co", "co.", "nv", "n.v.", "sa", "s.a.",
    "se", "ag", "ordinary", "registered", "shares", "the", "and", "of",
    "com", "technologies", "technology", "systems", "properties",
}


def _first_token(name: str) -> str:
    """Return the first significant word in a company name, lowercased."""
    # Drop everything after a parenthesis
    clean = re.split(r"[(\-]", name)[0]
    words = re.split(r"[\s,\.]+", clean.lower())
    words = [w for w in words if w and w not in _NAME_STOP]
    return words[0] if words else ""


def _tag_article_tickers(article: dict, positions: list[dict]) -> list[str]:
    """Return tickers from positions whose name appears in the article."""
    text = (article.get("headline", "") + " " + article.get("description", "")).lower()
    matched: list[str] = []
    for p in positions:
        name = p["name"]
        ticker = p["ticker"]
        full = re.split(r"[(\-]", name)[0].strip().lower()
        if len(full) >= 5 and full in text:
            matched.append(ticker)
            continue
        tok = _first_token(name)
        if tok and len(tok) >= 4 and re.search(r"\b" + re.escape(tok) + r"\b", text):
            matched.append(ticker)
    # Dedupe, cap at 3
    seen: set[str] = set()
    out: list[str] = []
    for t in matched:
        if t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) >= 3:
                break
    return out


def fetch_portfolio_news(positions: list[dict]) -> list[dict]:
    """Fetch the 10 most relevant recent articles and tag with tickers."""
    api_key = os.getenv("NEWS_API_KEY", "")
    newsapi = NewsApiClient(api_key=api_key)

    names = [p["name"] for p in positions]

    suffix = " AND (stock OR earnings OR revenue OR market)"
    max_name_chars = 500 - len("(") - len(")") - len(suffix)
    parts: list[str] = []
    used = 0
    for name in names:
        clean = name.replace("&", "and").replace("(", "").replace(")", "").replace('"', "").strip()
        if not clean:
            continue
        token = f'"{clean}"'
        joiner_len = len(" OR ") if parts else 0
        if used + joiner_len + len(token) > max_name_chars:
            break
        parts.append(token)
        used += joiner_len + len(token)

    name_clause = " OR ".join(parts)
    query = f"({name_clause}){suffix}"

    try:
        resp = newsapi.get_everything(
            q=query, language="en", sort_by="relevancy", page_size=30,
        )
        raw = resp.get("articles", [])
        out: list[dict] = []
        seen: set[str] = set()
        for a in raw:
            title = (a.get("title") or "").strip()
            if not title or title == "[Removed]" or title in seen:
                continue
            seen.add(title)
            article = dict(
                headline=title,
                description=a.get("description") or "",
                source=a.get("source", {}).get("name", "Unknown"),
                date=(a.get("publishedAt") or "")[:10],
                url=a.get("url", ""),
            )
            article["tickers"] = _tag_article_tickers(article, positions)
            out.append(article)
            if len(out) >= 10:
                break
        log.info("Fetched %d portfolio news articles", len(out))
        return out
    except Exception as exc:
        log.error("Failed to fetch portfolio news: %s", exc)
        return []


_MACRO_KEYWORDS = [
    "fed", "inflation", "rate", "rates", "gdp", "election", "policy",
    "regulation", "tariff", "trade", "oil", "war", "treaty", "ai",
    "breakthrough", "discovery", "vaccine", "climate",
]
_EXCLUDE_KEYWORDS = [
    "airline", "airlines", "podcast", "podcasts", "celebrity", "sport", "sports",
]
_PREMIUM_SOURCES = (
    "reuters,bloomberg,the-wall-street-journal,"
    "financial-times,the-economist,bbc-news,associated-press"
)


def _matches_macro(title: str) -> bool:
    t = title.lower()
    if any(re.search(r"\b" + re.escape(k) + r"\b", t) for k in _EXCLUDE_KEYWORDS):
        return False
    return any(re.search(r"\b" + re.escape(k) + r"\b", t) for k in _MACRO_KEYWORDS)


def _classify_news(title: str) -> str:
    t = title.lower()
    politics_kw = [
        "election", "president", "senator", "government", "policy", "regulation",
        "tariff", "war", "treaty", "military", "sanction", "congress", "vote",
        "parliament", "ukraine", "gaza", "nato", "putin", "biden", "trump",
    ]
    science_kw = [
        "ai", "breakthrough", "discovery", "vaccine", "climate", "research",
        "scientist", "satellite", "quantum", "study", "nasa", "spacex",
    ]
    economy_kw = [
        "fed", "inflation", "rate", "gdp", "trade", "oil", "market", "economy",
        "bank", "dollar", "euro", "recession", "growth", "stock",
    ]
    p = sum(1 for k in politics_kw if re.search(r"\b" + k + r"\b", t))
    s = sum(1 for k in science_kw  if re.search(r"\b" + k + r"\b", t))
    e = sum(1 for k in economy_kw  if re.search(r"\b" + k + r"\b", t))
    if p >= s and p >= e and p > 0:
        return "POLITICS"
    if s > e and s > 0:
        return "SCIENCE"
    return "ECONOMY"


def fetch_general_news() -> list[dict]:
    """Macro-relevant headlines from premium sources, filtered for substance."""
    api_key = os.getenv("NEWS_API_KEY", "")
    newsapi = NewsApiClient(api_key=api_key)

    query = " OR ".join(_MACRO_KEYWORDS)

    def _collect(resp_articles, existing_titles):
        out: list[dict] = []
        for a in resp_articles:
            title = (a.get("title") or "").strip()
            if not title or title == "[Removed]" or title in existing_titles:
                continue
            if not _matches_macro(title):
                continue
            existing_titles.add(title)
            out.append(dict(
                headline=title,
                description=a.get("description") or "",
                source=a.get("source", {}).get("name", "Unknown"),
                date=(a.get("publishedAt") or "")[:10],
                url=a.get("url", ""),
                category=_classify_news(title),
            ))
        return out

    seen: set[str] = set()
    articles: list[dict] = []

    # Premium sources first
    try:
        resp = newsapi.get_everything(
            q=query,
            sources=_PREMIUM_SOURCES,
            language="en",
            sort_by="relevancy",
            page_size=40,
        )
        articles.extend(_collect(resp.get("articles", []), seen))
    except Exception as exc:
        log.warning("Premium-source query failed (%s) — falling back", exc)

    # Fallback to all sources if we came up short
    if len(articles) < 5:
        try:
            resp = newsapi.get_everything(
                q=query, language="en", sort_by="relevancy", page_size=40,
            )
            articles.extend(_collect(resp.get("articles", []), seen))
        except Exception as exc:
            log.error("Failed to fetch general news: %s", exc)

    articles = articles[:5]
    log.info("Fetched %d general news articles", len(articles))
    return articles


# ── Claude analysis ───────────────────────────────────────────────────────────
def _build_user_context(positions, summary, benchmarks, port_news, gen_news, metrics) -> str:
    positions_text = "\n".join(
        f"- {p['name']} ({p['ticker']}): {p['shares']:.4g} shares | "
        f"current €{p['price_eur']:.2f} | "
        f"value €{p['value_eur']:,.2f} | P&L €{p['pl_eur']:+,.2f} ({p['pl_pct']:+.2f}%) | "
        f"1W {p['weekly_pct']:+.2f}% 1M {p['monthly_pct']:+.2f}% YTD {p['ytd_pct']:+.2f}%"
        for p in positions
    )
    bench_text = "\n".join(
        f"- {b['name']}: 1W {b['weekly_pct']:+.2f}% 1M {b['monthly_pct']:+.2f}% YTD {b['ytd_pct']:+.2f}%"
        for b in benchmarks
    )
    port_news_text = "\n".join(
        f"- [{a['headline']}]({a['url']}) — {a['source']}, {a['date']}"
        for a in port_news
    )
    gen_news_text = "\n".join(
        f"- [{a['headline']}]({a['url']}) — {a['source']}, {a['date']}"
        for a in gen_news
    )
    metrics_text = ""
    if metrics:
        metrics_text = (
            f"\nADVANCED METRICS\n"
            f"Sharpe Ratio: {metrics.get('sharpe', 0):.2f} | "
            f"Sortino Ratio: {metrics.get('sortino', 0):.2f}\n"
            f"Beta vs S&P 500: {metrics.get('beta', 0):.2f} | "
            f"Alpha (annualized): {metrics.get('alpha', 0)*100:+.2f}%\n"
            f"Annualized Volatility: {metrics.get('volatility', 0)*100:.1f}% | "
            f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%\n"
            f"Calmar Ratio: {metrics.get('calmar', 0):.2f} | "
            f"VaR 95% (1-day): {metrics.get('var_95', 0)*100:.2f}%\n"
            f"Correlation to S&P 500: {metrics.get('correlation', 0):.2f} | "
            f"HHI Concentration: {metrics.get('hhi', 0):.4f}\n"
            f"Win Rate: {metrics.get('win_rate', 0):.1f}%"
        )
    return (
        f"PORTFOLIO SUMMARY\n"
        f"Total value: €{summary['total_value_eur']:,.2f} | "
        f"Total P&L: €{summary['total_pl_eur']:+,.2f} ({summary['total_pl_pct']:+.2f}%)\n"
        f"Weekly: {summary['weekly_pct']:+.2f}% | Monthly: {summary['monthly_pct']:+.2f}% | "
        f"YTD: {summary['ytd_pct']:+.2f}%\n\n"
        f"POSITIONS\n{positions_text}\n\n"
        f"BENCHMARKS\n{bench_text}"
        f"{metrics_text}\n\n"
        f"PORTFOLIO NEWS\n{port_news_text}\n\n"
        f"GENERAL NEWS\n{gen_news_text}"
    )


def call_claude(positions, summary, benchmarks, port_news, gen_news, metrics) -> str:
    """Return Claude's analysis as markdown (sections 1-4)."""
    client = anthropic.Anthropic()

    system_prompt = (
        "You are a personal financial analyst writing for a private wealth briefing. "
        "Your tone is editorial, precise, confident — like the Financial Times or The "
        "Economist. You have access to the user's full portfolio data, news, benchmarks, "
        "and advanced quant metrics including Sharpe, Beta, Alpha, Sortino, VaR, Max "
        "Drawdown. Use these metrics to give specific analysis. Write a structured report "
        "with exactly these four sections:\n\n"
        "## (1) Portfolio Performance Summary — key numbers, what's up, what's down.\n"
        "## (2) Benchmark Comparison — how the portfolio did vs S&P 500, Nasdaq, and MSCI "
        "World. Reference Alpha and Beta where relevant.\n"
        "## (3) News & Market Context — the 3 most important implications of the news for "
        "this specific portfolio.\n"
        "## (4) Actionable Suggestions — 3 to 5 specific recommendations. Format EACH item "
        "on its own line as:\n"
        "[ACTION] Title — Detail sentence.\n"
        "where ACTION is EXACTLY one of: TRIM, EXIT, RESEARCH, ADD. Do not use bullets or "
        "numbers — just the [ACTION] prefix. Example:\n"
        "[TRIM] Reduce NVDA concentration — At 22% of portfolio, HHI is elevated; "
        "trimming to 15% would still capture upside while diversifying tail risk.\n\n"
        "Use **bold** sparingly for headline figures. Be direct and concise. No disclaimers."
    )

    user_content = _build_user_context(positions, summary, benchmarks, port_news, gen_news, metrics)

    log.info("Calling Claude API for analysis (model: %s)…", CLAUDE_MODEL)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_content}],
    )
    log.info(
        "Claude analysis received (input tokens: %d, cached: %d)",
        response.usage.input_tokens,
        response.usage.cache_read_input_tokens or 0,
    )
    return response.content[0].text


def call_claude_headline(summary: dict, benchmarks: list[dict], metrics: dict) -> str:
    """Return a single-sentence editorial headline (<=20 words, one number)."""
    client = anthropic.Anthropic()

    system = (
        "You write a single-sentence editorial headline for a fortnightly portfolio "
        "briefing, styled like the Financial Times. Requirements: max 20 words; MUST "
        "contain at least one specific number, percentage, or comparison; subtly "
        "insightful; no filler. Output ONLY the sentence — no quotes, no preamble."
    )

    bench_lines = "\n".join(
        f"{b['name']}: monthly {b['monthly_pct']:+.2f}%, YTD {b['ytd_pct']:+.2f}%"
        for b in benchmarks
    )
    metrics_line = ""
    if metrics:
        metrics_line = (
            f"Sharpe {metrics.get('sharpe', 0):.2f}, "
            f"Alpha {metrics.get('alpha', 0)*100:+.2f}%, "
            f"Beta {metrics.get('beta', 0):.2f}"
        )
    context = (
        f"Portfolio monthly: {summary['monthly_pct']:+.2f}%\n"
        f"Portfolio YTD: {summary['ytd_pct']:+.2f}%\n"
        f"Total P&L: €{summary['total_pl_eur']:+,.0f} ({summary['total_pl_pct']:+.2f}%)\n"
        f"{bench_lines}\n"
        f"{metrics_line}"
    )

    log.info("Calling Claude API for headline…")
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=120,
        system=[{"type": "text", "text": system}],
        messages=[{"role": "user", "content": context}],
    )
    text = response.content[0].text.strip()
    # Strip surrounding quotes if Claude added any
    text = text.strip('"').strip("'").strip()
    # Keep to one line
    text = text.splitlines()[0].strip() if text else ""
    return text or "Your portfolio held steady this fortnight."


# ── Analysis / actionable parsing ─────────────────────────────────────────────
def split_analysis(text: str) -> tuple[str, str]:
    """Return (body_markdown_sections_1_to_3, actionable_raw_text)."""
    parts = re.split(r"\n?##\s+", "\n" + text)
    body_parts: list[str] = []
    actionable_text = ""
    for sec in parts[1:]:
        lines = sec.splitlines()
        if not lines:
            continue
        heading = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        heading_lower = heading.lower()
        if "(4)" in heading_lower or "action" in heading_lower:
            actionable_text = body
        else:
            body_parts.append(f"## {heading}\n{body}")
    return "\n\n".join(body_parts), actionable_text


def parse_actionable(text: str) -> list[dict]:
    """Parse `[ACTION] Title — Detail` lines into structured items."""
    if not text:
        return []
    pattern = re.compile(r"\[(TRIM|EXIT|RESEARCH|ADD)\]\s*:?\s*(.+)", re.IGNORECASE)
    items: list[dict] = []
    current: dict | None = None
    for raw in text.splitlines():
        line = re.sub(r"^[-*•\d\.\)]+\s*", "", raw).strip()
        line = re.sub(r"^\*\*(.+?)\*\*:?", r"\1", line)
        m = pattern.match(line)
        if m:
            if current:
                items.append(current)
            action = m.group(1).upper()
            rest = m.group(2).strip().strip("*").strip()
            split_m = re.match(r"(.+?)\s*[—–-]\s*(.+)", rest)
            if split_m:
                title = split_m.group(1).strip().rstrip(":")
                detail = split_m.group(2).strip()
            else:
                title, detail = rest, ""
            current = dict(action=action, title=title, detail=detail)
        elif current and line:
            # Continuation
            if current["detail"]:
                current["detail"] += " " + line
            else:
                current["detail"] = line
    if current:
        items.append(current)
    return items[:5]


# ── Metric interpretations ────────────────────────────────────────────────────
def interp_sharpe(v):
    if v < 0:   return "Negative risk-adjusted return"
    if v < 0.5: return "Below-average risk-adjusted return"
    if v < 1.0: return "Acceptable risk-adjusted return"
    if v < 2.0: return "Good risk-adjusted return"
    return "Excellent risk-adjusted return"


def interp_sortino(v):
    if v < 0:   return "Downside drag outweighs gains"
    if v < 0.5: return "Downside risk dominates"
    if v < 1.0: return "Acceptable downside-adjusted return"
    return "Strong downside-adjusted return"


def interp_beta(v):
    if v < 0.5: return "Much less volatile than market"
    if v < 0.8: return "Less volatile than the market"
    if v < 1.2: return "Moves in line with the market"
    if v < 1.5: return "More volatile than the market"
    return "Significantly amplifies market moves"


def interp_alpha(v_pct):
    direction = "Out" if v_pct >= 0 else "Under"
    return f"{direction}performs the market by {abs(v_pct):.1f}%/yr risk-adjusted"


def interp_vol(v):
    pct = v * 100
    if pct < 10: return "Low portfolio volatility"
    if pct < 20: return "Moderate portfolio volatility"
    if pct < 30: return "High portfolio volatility"
    return "Very high portfolio volatility"


def interp_dd(v):
    pct = abs(v) * 100
    if pct < 10: return "Limited drawdown over the year"
    if pct < 20: return "Moderate drawdown"
    if pct < 30: return "Significant drawdown"
    return "Severe drawdown"


def interp_calmar(v):
    if v < 0:   return "Annual return is negative"
    if v < 0.5: return "Poor return vs drawdown risk"
    if v < 1.0: return "Acceptable return-to-drawdown"
    return "Strong return vs drawdown risk"


def interp_var(v):
    return f"Daily loss stays below {abs(v)*100:.2f}% on 95% of days"


def interp_corr(v):
    if v < 0.3: return "Low market correlation"
    if v < 0.6: return "Moderate market correlation"
    if v < 0.8: return "Fairly correlated to market"
    return "Highly correlated to market"


def interp_hhi(v):
    if v < 0.10: return "Well diversified"
    if v < 0.18: return "Moderately concentrated"
    if v < 0.25: return "Top positions dominate"
    return "Highly concentrated"


def interp_wr(v):
    if v < 40: return "Most positions underwater"
    if v < 60: return "Roughly half profitable"
    if v < 80: return "Most positions profitable"
    return "Strong profitable mix"


# ── HTML helpers ──────────────────────────────────────────────────────────────
def _color(val: float) -> str:
    return SAGE if val >= 0 else BURGUNDY


def _fmt(val: float) -> str:
    return f"{val:+.2f}%"


def _markdown_to_html(text: str) -> str:
    """Convert Claude's markdown output to clean editorial HTML."""
    html_parts: list[str] = []
    in_ul = False
    in_p = False

    def close_p():
        nonlocal in_p
        if in_p:
            html_parts.append("</p>")
            in_p = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            html_parts.append("</ul>")
            in_ul = False

    def inline(s: str) -> str:
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                   rf'<a href="\2" style="color:{NAVY};text-decoration:none">\1</a>', s)
        return s

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            close_ul()
            close_p()
            continue
        if stripped.startswith("### "):
            close_ul(); close_p()
            html_parts.append(
                f'<h4 style="font-family:Georgia,serif;font-size:15px;color:{TEXT_PRIMARY};'
                f'margin:18px 0 6px;font-weight:600">{inline(stripped[4:].strip())}</h4>'
            )
        elif stripped.startswith("## "):
            close_ul(); close_p()
            html_parts.append(
                f'<h3 style="font-family:Georgia,serif;font-size:17px;color:{DARK_GOLD};'
                f'margin:24px 0 10px;font-weight:normal;font-style:italic">'
                f'{inline(stripped[3:].strip())}</h3>'
            )
        elif re.match(r"^[-*]\s+", stripped):
            close_p()
            if not in_ul:
                html_parts.append(
                    '<ul style="margin:8px 0 14px;padding-left:22px;list-style-type:disc">'
                )
                in_ul = True
            item = re.sub(r"^[-*]\s+", "", stripped)
            html_parts.append(
                f'<li style="margin-bottom:6px;line-height:1.65">{inline(item)}</li>'
            )
        else:
            close_ul()
            if not in_p:
                html_parts.append('<p style="margin:0 0 14px;line-height:1.7">')
                in_p = True
            html_parts.append(inline(stripped) + " ")

    close_ul()
    close_p()
    return "\n".join(html_parts)


def _analysis_with_drop_cap(html: str) -> str:
    """Insert a gold drop-cap on the first letter of the first <p>."""
    def repl(m):
        open_tag = m.group(1)
        content = m.group(2)
        if not content.strip():
            return m.group(0)
        letter_match = re.search(r"([A-Za-z])", content)
        if not letter_match:
            return m.group(0)
        idx = letter_match.start()
        before = content[:idx]
        letter = content[idx]
        after = content[idx + 1:]
        drop = (
            f'<span style="float:left;font-family:Georgia,serif;font-size:42px;'
            f'color:{GOLD};line-height:0.9;padding:6px 10px 0 0">{letter}</span>'
        )
        return f"{open_tag}{before}{drop}{after}</p>"

    return re.sub(r"(<p[^>]*>)(.*?)</p>", repl, html, count=1, flags=re.DOTALL)


def _label_cell(label: str) -> str:
    return (
        f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
        f'text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};'
        f'margin-bottom:6px">{label}</div>'
    )


def _sect_header(num: str, label: str) -> str:
    return (
        f'<h2 style="font-family:Georgia,serif;font-size:22px;font-weight:normal;'
        f'color:{TEXT_PRIMARY};border-bottom:2px solid {TEXT_PRIMARY};'
        f'padding-bottom:10px;margin:0 0 24px">'
        f'<span style="color:{DARK_GOLD}">{num}.</span> {label}'
        f'</h2>'
    )


def _fig_caption(n: int, label: str, cid: str) -> str:
    return (
        f'<div style="margin:28px 0 24px">'
        f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
        f'text-transform:uppercase;letter-spacing:2px;color:{DARK_GOLD};'
        f'margin-bottom:10px">Fig. {n} &middot; {label}</div>'
        f'<img src="cid:{cid}" width="560" alt="{label}" '
        f'style="display:block;max-width:100%;height:auto;border:0">'
        f'</div>'
    )


def _emphasize_numbers(text: str) -> str:
    """Wrap first number/percentage in gold italic for the headline."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    replaced = [False]

    def repl(m):
        if replaced[0]:
            return m.group(0)
        replaced[0] = True
        return (
            f'<em style="color:{GOLD};font-style:italic;'
            f'font-variant-numeric:tabular-nums">{m.group(0)}</em>'
        )

    # Match numbers with optional sign/decimal/% or points
    return re.sub(r"[+-]?\d+(?:[.,]\d+)?(?:%|\s?(?:points?|pts|bps|ppt))?", repl, text)


# ── HTML email builder ────────────────────────────────────────────────────────
def build_html_email(
    positions: list[dict],
    summary: dict,
    benchmarks: list[dict],
    port_news: list[dict],
    gen_news: list[dict],
    analysis_body: str,
    actionable_items: list[dict],
    headline: str,
    chart_cids: dict[str, str],
    metrics: dict,
    issue_num: int,
) -> str:
    today = date.today()
    today_str = today.strftime("%B %-d, %Y") if os.name != "nt" else today.strftime("%B %d, %Y").replace(" 0", " ")

    total = summary["total_value_eur"]
    fortnight_pct = summary["monthly_pct"]
    denom = 100.0 + fortnight_pct
    fortnight_pl = total * fortnight_pct / denom if denom else 0.0
    fortnight_color = _color(fortnight_pl)

    # ── Headline ──────────────────────────────────────────────────────────────
    headline_html = _emphasize_numbers(headline)

    # ── Primary metric strip ──────────────────────────────────────────────────
    primary_data = [
        ("Sharpe", f"{metrics.get('sharpe', 0):.2f}", interp_sharpe(metrics.get("sharpe", 0))),
        ("Beta",   f"{metrics.get('beta', 0):.2f}",   interp_beta(metrics.get("beta", 0))),
        ("Alpha",  f"{metrics.get('alpha', 0)*100:+.1f}%", interp_alpha(metrics.get("alpha", 0)*100)),
        ("Max DD", f"{metrics.get('max_drawdown', 0)*100:.1f}%", interp_dd(metrics.get("max_drawdown", 0))),
    ]
    primary_cells = ""
    for i, (label, value, interp) in enumerate(primary_data):
        left_border = f"border-left:1px solid {RULE_FAINT};" if i > 0 else ""
        primary_cells += (
            f'<td style="text-align:center;padding:22px 10px;{left_border}width:25%;vertical-align:top">'
            f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};'
            f'margin-bottom:10px">{label}</div>'
            f'<div style="font-family:Georgia,serif;font-size:26px;color:{TEXT_PRIMARY};'
            f'margin-bottom:8px;font-variant-numeric:tabular-nums">{value}</div>'
            f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'color:{TEXT_SECONDARY};line-height:1.4">{interp}</div>'
            f'</td>'
        )

    # ── Secondary metric strip (4 across) ─────────────────────────────────────
    secondary = [
        ("Sortino",    f"{metrics.get('sortino', 0):.2f}",       interp_sortino(metrics.get("sortino", 0))),
        ("Calmar",     f"{metrics.get('calmar', 0):.2f}",        interp_calmar(metrics.get("calmar", 0))),
        ("Volatility", f"{metrics.get('volatility', 0)*100:.1f}%", interp_vol(metrics.get("volatility", 0))),
        ("VaR 95%",    f"{metrics.get('var_95', 0)*100:.2f}%",   interp_var(metrics.get("var_95", 0))),
        ("Correlation", f"{metrics.get('correlation', 0):.2f}",  interp_corr(metrics.get("correlation", 0))),
        ("HHI",        f"{metrics.get('hhi', 0):.3f}",           interp_hhi(metrics.get("hhi", 0))),
        ("Win Rate",   f"{metrics.get('win_rate', 0):.0f}%",     interp_wr(metrics.get("win_rate", 0))),
    ]
    # Pad to multiple of 4
    while len(secondary) % 4 != 0:
        secondary.append(("", "", ""))

    sec_rows = ""
    for i in range(0, len(secondary), 4):
        sec_rows += "<tr>"
        for j in range(4):
            label, value, interp = secondary[i + j]
            if not label:
                sec_rows += '<td style="width:25%"></td>'
                continue
            sec_rows += (
                f'<td style="padding:16px 12px;border-bottom:1px solid {RULE_FAINT};'
                f'vertical-align:top;width:25%">'
                f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:9px;'
                f'text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};'
                f'margin-bottom:4px">{label}</div>'
                f'<div style="font-family:Georgia,serif;font-size:18px;color:{TEXT_PRIMARY};'
                f'font-variant-numeric:tabular-nums">{value}</div>'
                f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
                f'color:{TEXT_SECONDARY};margin-top:3px;line-height:1.35">{interp}</div>'
                f'</td>'
            )
        sec_rows += "</tr>"

    # ── Positions table ───────────────────────────────────────────────────────
    def _th(label: str, align: str = "left") -> str:
        return (
            f'<th style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};'
            f'text-align:{align};padding:10px 4px;border-bottom:1px solid {TEXT_PRIMARY};'
            f'font-weight:600">{label}</th>'
        )

    pos_rows_html = ""
    for p in positions:
        pos_rows_html += (
            f'<tr>'
            f'<td style="padding:12px 4px;border-bottom:1px solid {RULE_FAINT};'
            f'font-family:Georgia,serif;font-size:14px">'
            f'{p["name"]}'
            f' <span style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'color:{TEXT_SECONDARY};letter-spacing:1px;margin-left:6px">{p["ticker"]}</span>'
            f'</td>'
            f'<td style="padding:12px 4px;border-bottom:1px solid {RULE_FAINT};'
            f'font-family:Arial,Helvetica,sans-serif;font-size:13px;text-align:right;'
            f'font-variant-numeric:tabular-nums;color:{TEXT_PRIMARY}">{p["shares"]:.2f}</td>'
            f'<td style="padding:12px 4px;border-bottom:1px solid {RULE_FAINT};'
            f'font-family:Arial,Helvetica,sans-serif;font-size:13px;text-align:right;'
            f'font-variant-numeric:tabular-nums;color:{TEXT_PRIMARY}">€{p["value_eur"]:,.0f}</td>'
            f'<td style="padding:12px 4px;border-bottom:1px solid {RULE_FAINT};'
            f'font-family:Arial,Helvetica,sans-serif;font-size:13px;text-align:right;'
            f'font-variant-numeric:tabular-nums;color:{_color(p["pl_pct"])}">{_fmt(p["pl_pct"])}</td>'
            f'<td style="padding:12px 4px;border-bottom:1px solid {RULE_FAINT};'
            f'font-family:Arial,Helvetica,sans-serif;font-size:13px;text-align:right;'
            f'font-variant-numeric:tabular-nums;color:{_color(p["ytd_pct"])}">{_fmt(p["ytd_pct"])}</td>'
            f'</tr>'
        )

    # ── By the Numbers (charts) ───────────────────────────────────────────────
    fig_order = [
        ("portfolio_pie", "Portfolio Allocation"),
        ("sector_pie",    "Sector Allocation"),
        ("geo_pie",       "Geographic Allocation"),
        ("movers_bar",    "Top & Bottom Movers (YTD)"),
    ]
    figs_html = ""
    n = 1
    for key, label in fig_order:
        if key in chart_cids:
            figs_html += _fig_caption(n, label, chart_cids[key])
            n += 1

    # ── The Analysis ──────────────────────────────────────────────────────────
    analysis_html = _markdown_to_html(analysis_body)
    analysis_html = _analysis_with_drop_cap(analysis_html)

    # ── Portfolio News (numbered 01-10) ───────────────────────────────────────
    port_news_html = ""
    for i, a in enumerate(port_news, 1):
        tickers_html = ""
        if a.get("tickers"):
            tickers_html = " &middot; " + " ".join(
                f'<span style="color:{NAVY};font-family:Arial,Helvetica,sans-serif;'
                f'font-size:10px;font-weight:700;letter-spacing:1.5px">{t}</span>'
                for t in a["tickers"]
            )
        port_news_html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" '
            f'style="margin-bottom:18px;border-bottom:1px solid {RULE_FAINT}">'
            f'<tr>'
            f'<td style="width:42px;vertical-align:top;padding:0 0 16px">'
            f'<div style="font-family:Georgia,serif;font-size:20px;color:{GOLD}">{i:02d}</div>'
            f'</td>'
            f'<td style="vertical-align:top;padding:0 0 16px">'
            f'<a href="{a["url"]}" style="text-decoration:none;color:{TEXT_PRIMARY}">'
            f'<div style="font-family:Georgia,serif;font-size:15px;line-height:1.45;'
            f'margin-bottom:8px">{a["headline"]}</div>'
            f'</a>'
            f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'color:{TEXT_SECONDARY};text-transform:uppercase;letter-spacing:1.5px">'
            f'{a["source"]} &middot; {a["date"]}{tickers_html}'
            f'</div>'
            f'</td>'
            f'</tr>'
            f'</table>'
        )
    if not port_news_html:
        port_news_html = (
            f'<p style="font-family:Georgia,serif;color:{TEXT_SECONDARY};font-style:italic">'
            f'No portfolio news available this fortnight.</p>'
        )

    # ── The World (5 general) ─────────────────────────────────────────────────
    cat_color = {"POLITICS": BURGUNDY, "ECONOMY": SAGE, "SCIENCE": NAVY}
    gen_news_html = ""
    for a in gen_news:
        cat = a.get("category", "ECONOMY")
        color = cat_color.get(cat, SAGE)
        gen_news_html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" '
            f'style="margin-bottom:22px;border-bottom:1px solid {RULE_FAINT}">'
            f'<tr>'
            f'<td style="width:90px;vertical-align:top;padding:2px 16px 18px 0">'
            f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'color:{color};text-transform:uppercase;letter-spacing:2px;font-weight:700">'
            f'{cat}</div>'
            f'</td>'
            f'<td style="vertical-align:top;padding:0 0 18px">'
            f'<a href="{a["url"]}" style="text-decoration:none;color:{TEXT_PRIMARY}">'
            f'<div style="font-family:Georgia,serif;font-size:15px;line-height:1.45;'
            f'margin-bottom:8px">{a["headline"]}</div>'
            f'</a>'
            f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'color:{TEXT_SECONDARY};text-transform:uppercase;letter-spacing:1.5px">'
            f'{a["source"]} &middot; {a["date"]}'
            f'</div>'
            f'</td>'
            f'</tr>'
            f'</table>'
        )
    if not gen_news_html:
        gen_news_html = (
            f'<p style="font-family:Georgia,serif;color:{TEXT_SECONDARY};font-style:italic">'
            f'No general news available this fortnight.</p>'
        )

    # ── Actionable ────────────────────────────────────────────────────────────
    action_color_map = {"TRIM": SAGE, "EXIT": BURGUNDY, "RESEARCH": NAVY, "ADD": GOLD}
    actionable_html = ""
    for i, item in enumerate(actionable_items, 1):
        action = item.get("action", "RESEARCH")
        a_color = action_color_map.get(action, TEXT_PRIMARY)
        title = item.get("title", "")
        detail = item.get("detail", "")
        actionable_html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" '
            f'style="margin-bottom:28px;border-bottom:1px solid {RULE_FAINT}">'
            f'<tr>'
            f'<td style="width:64px;vertical-align:top;padding:6px 18px 22px 0">'
            f'<div style="font-family:Georgia,serif;font-size:28px;color:{GOLD};'
            f'font-variant-numeric:tabular-nums">{i:02d}</div>'
            f'</td>'
            f'<td style="vertical-align:top;padding:6px 0 22px">'
            f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'color:{a_color};text-transform:uppercase;letter-spacing:2px;'
            f'font-weight:700;margin-bottom:8px">{action}</div>'
            f'<div style="font-family:Georgia,serif;font-size:16px;color:{TEXT_PRIMARY};'
            f'margin-bottom:8px;line-height:1.35">{title}</div>'
            f'<div style="font-family:Georgia,serif;font-size:14px;color:{TEXT_PRIMARY};'
            f'line-height:1.65">{detail}</div>'
            f'</td>'
            f'</tr>'
            f'</table>'
        )
    if not actionable_html:
        actionable_html = (
            f'<p style="font-family:Georgia,serif;color:{TEXT_SECONDARY};font-style:italic">'
            f'No specific actions recommended this fortnight.</p>'
        )

    # ── Full HTML ─────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Fortnight &middot; No. {issue_num}</title>
</head>
<body style="margin:0;padding:0;background:{BG_CREAM};font-family:Georgia,serif;color:{TEXT_PRIMARY}">
<div style="max-width:640px;margin:0 auto;background:{BG_CREAM}">

<!-- Masthead -->
<table width="100%" cellpadding="0" cellspacing="0" style="background:{CHARCOAL};border-bottom:3px solid {GOLD}">
<tr><td style="padding:40px">
  <table width="100%" cellpadding="0" cellspacing="0"><tr>
    <td style="font-family:Arial,Helvetica,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY}">Portfolio Report &middot; No. {issue_num}</td>
    <td style="font-family:Arial,Helvetica,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};text-align:right">{today_str}</td>
  </tr></table>
  <h1 style="font-family:Georgia,serif;font-size:38px;font-weight:normal;margin:28px 0 6px;color:{BG_CREAM};letter-spacing:-0.5px">The Fortnight</h1>
  <p style="font-family:Georgia,serif;font-style:italic;color:{GOLD};margin:0 0 36px;font-size:14px">A private wealth briefing</p>
  <table width="100%" cellpadding="0" cellspacing="0"><tr>
    <td style="padding-right:32px;width:50%;vertical-align:top">
      <div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};margin-bottom:8px">Total Value</div>
      <div style="font-family:Georgia,serif;font-size:32px;color:{BG_CREAM};font-variant-numeric:tabular-nums">€{total:,.0f}</div>
    </td>
    <td style="width:50%;vertical-align:top">
      <div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};margin-bottom:8px">Fortnight P&amp;L</div>
      <div style="font-family:Georgia,serif;font-size:32px;color:{fortnight_color};font-variant-numeric:tabular-nums">€{fortnight_pl:+,.0f} <span style="font-size:18px;color:{TEXT_SECONDARY}">({fortnight_pct:+.2f}%)</span></div>
    </td>
  </tr></table>
</td></tr>
</table>

<!-- The Headline -->
<div style="padding:48px 40px 24px">
  <p style="font-family:Georgia,serif;font-size:22px;line-height:1.45;color:{TEXT_PRIMARY};margin:0">{headline_html}</p>
</div>

<!-- Primary metric strip -->
<div style="padding:8px 40px">
  <table width="100%" cellpadding="0" cellspacing="0" style="border-top:1px solid {TEXT_PRIMARY};border-bottom:1px solid {TEXT_PRIMARY}">
    <tr>{primary_cells}</tr>
  </table>
</div>

<!-- Secondary metric strip -->
<div style="padding:4px 40px 24px">
  <table width="100%" cellpadding="0" cellspacing="0">{sec_rows}</table>
</div>

<!-- Section I. Positions -->
<div style="padding:24px 40px">
  {_sect_header("I", "Positions")}
  <table width="100%" cellpadding="0" cellspacing="0">
    <thead><tr>
      {_th("Holding", "left")}
      {_th("Shares", "right")}
      {_th("Value", "right")}
      {_th("P&amp;L", "right")}
      {_th("YTD", "right")}
    </tr></thead>
    <tbody>{pos_rows_html}</tbody>
  </table>
</div>

<!-- Section II. By the Numbers -->
<div style="padding:24px 40px">
  {_sect_header("II", "By the Numbers")}
  {figs_html}
</div>

<!-- Section III. The Analysis -->
<div style="padding:24px 40px">
  {_sect_header("III", "The Analysis")}
  <div style="font-family:Georgia,serif;font-size:15px;line-height:1.7;color:{TEXT_PRIMARY}">
    {analysis_html}
  </div>
</div>

<!-- Section IV. Portfolio News -->
<div style="padding:24px 40px">
  {_sect_header("IV", "Portfolio News")}
  {port_news_html}
</div>

<!-- Section V. The World -->
<div style="padding:24px 40px">
  {_sect_header("V", "The World")}
  {gen_news_html}
</div>

<!-- Section VI. Actionable -->
<div style="padding:24px 40px 48px">
  {_sect_header("VI", "Actionable")}
  {actionable_html}
</div>

<!-- Footer -->
<table width="100%" cellpadding="0" cellspacing="0" style="background:{CHARCOAL};border-top:2px solid {GOLD}">
<tr><td style="padding:32px 40px;text-align:center;font-family:Arial,Helvetica,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY}">
The Fortnight &middot; No. {issue_num} &middot; {today_str}
</td></tr>
</table>

</div>
</body>
</html>"""


# ── Email sender ──────────────────────────────────────────────────────────────
def send_email(html: str, chart_files: dict[str, str]) -> None:
    gmail = os.getenv("GMAIL_ADDRESS", "")
    password = os.getenv("GMAIL_APP_PASSWORD", "")
    recipient = os.getenv("RECIPIENT_EMAIL", "")

    if not all([gmail, password, recipient]):
        raise ValueError(
            "Missing email config. Set GMAIL_ADDRESS, GMAIL_APP_PASSWORD, "
            "and RECIPIENT_EMAIL in .env"
        )

    subject = f"The Fortnight — {date.today().strftime('%B %d, %Y')}"

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = gmail
    msg["To"] = recipient

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html, "html", "utf-8"))
    msg.attach(alt)

    for name, filepath in chart_files.items():
        try:
            with open(filepath, "rb") as f:
                img_data = f.read()
            img = MIMEImage(img_data, "png")
            cid = f"chart_{name}@portfolio"
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=f"{name}.png")
            msg.attach(img)
        except Exception as exc:
            log.warning("Could not attach chart %s: %s", name, exc)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail, password)
            server.sendmail(gmail, recipient, msg.as_string())
        log.info("Email sent to %s", recipient)
    finally:
        for filepath in chart_files.values():
            try:
                os.remove(filepath)
            except Exception:
                pass


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio Email Agent — runs on the second Saturday of each month"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the second-Saturday check and run immediately",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Build HTML and write to preview.html instead of sending email",
    )
    args = parser.parse_args()

    if not args.force and not is_second_saturday():
        msg = (
            "Today is not the second Saturday of the month. "
            "Use --force to run anyway."
        )
        log.info(msg)
        print(msg)
        return

    log.info("=== Portfolio Agent starting ===")

    try:
        ff_export = "investments.csv"
        if os.path.exists(ff_export):
            log.info("Found '%s' — refreshing portfolio.csv via import_finanzfluss…", ff_export)
            try:
                import import_finanzfluss
                count, warnings = import_finanzfluss.convert(
                    input_path=ff_export,
                    output_path="portfolio.csv",
                    verbose=False,
                )
                log.info("import_finanzfluss: wrote %d holdings to portfolio.csv", count)
                for w in warnings:
                    log.warning("import_finanzfluss: %s", w)
            except Exception as exc:
                log.error(
                    "import_finanzfluss failed (%s) — falling back to existing portfolio.csv",
                    exc,
                )

        rows = load_portfolio("portfolio.csv")

        log.info("Fetching FX rates…")
        fx_rates = get_fx_rates()

        log.info("Fetching portfolio market data…")
        positions, summary = fetch_portfolio_data(rows, fx_rates)
        positions.sort(key=lambda p: p["value_eur"], reverse=True)

        if not positions:
            log.error("No position data could be fetched. Aborting.")
            sys.exit(1)

        log.info("Fetching benchmark data…")
        benchmarks = fetch_benchmarks()

        log.info("Fetching portfolio news…")
        port_news = fetch_portfolio_news(positions)

        log.info("Fetching general news…")
        gen_news = fetch_general_news()

        log.info("Computing advanced metrics…")
        metrics = compute_advanced_metrics(positions, fx_rates)

        log.info("Generating charts…")
        charts = generate_charts(positions, summary, benchmarks)

        log.info("Calling Claude for analysis…")
        analysis_raw = call_claude(positions, summary, benchmarks, port_news, gen_news, metrics)
        analysis_body, actionable_raw = split_analysis(analysis_raw)
        actionable_items = parse_actionable(actionable_raw)

        log.info("Calling Claude for headline…")
        try:
            headline = call_claude_headline(summary, benchmarks, metrics)
        except Exception as exc:
            log.warning("Headline generation failed (%s) — using fallback", exc)
            headline = (
                f"Portfolio moved {summary['monthly_pct']:+.2f}% this fortnight, "
                f"finishing the year {summary['ytd_pct']:+.2f}% YTD."
            )

        log.info("Building HTML email…")
        chart_cids = {name: f"chart_{name}@portfolio" for name in charts}
        html = build_html_email(
            positions, summary, benchmarks, port_news, gen_news,
            analysis_body, actionable_items, headline,
            chart_cids, metrics, issue_number(),
        )

        if args.preview:
            with open("preview.html", "w", encoding="utf-8") as fh:
                fh.write(html)
            log.info("Preview written to preview.html (%d chart files in %s)",
                     len(charts), tempfile.gettempdir())
            print(f"Preview written to preview.html.")
            # Clean up temp charts
            for fp in charts.values():
                try:
                    os.remove(fp)
                except Exception:
                    pass
            return

        log.info("Sending email…")
        send_email(html, charts)

        log.info("=== Portfolio Agent completed successfully ===")
        print("Done! Report sent successfully.")

    except Exception:
        log.error("Agent failed:\n%s", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()




