"""Matplotlib charts for the email and the exposure breakdowns they draw."""
from __future__ import annotations

import logging
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from history import _read_history
from portfolio_io import _is_float
from theme import (
    _PALETTE,
    BG_CHART,
    BURGUNDY,
    DARK_GOLD,
    NAVY,
    SAGE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

log = logging.getLogger(__name__)


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


_HISTORY_CHART_MIN_ROWS = 4
_HISTORY_CHART_MARKER_MAX = 30


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

    dates    = [pd.Timestamp(r["date"]) for r in rows]
    port_raw = [float(r["port_index"]) for r in rows]
    sp_raw   = [float(r["sp500_eur"]) for r in rows]

    base_port = port_raw[0] or 1.0
    base_sp   = sp_raw[0]   or 1.0
    port_idx  = [v / base_port * 100 for v in port_raw]
    sp_idx    = [v / base_sp   * 100 for v in sp_raw]

    # Daily rows since the snapshot job: markers only while the series is short.
    few = len(rows) < _HISTORY_CHART_MARKER_MAX
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dates, port_idx, color=DARK_GOLD, linewidth=2,
            label="Portfolio", marker="o" if few else None, markersize=4)
    ax.plot(dates, sp_idx, color=NAVY, linewidth=1.5,
            linestyle="--", label="S&P 500 price index (EUR)", marker="o" if few else None,
            markersize=3)
    ax.axhline(100, color=TEXT_SECONDARY, linewidth=0.5, linestyle=":")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: mdates.num2date(x).strftime("%d %b '%y").lstrip("0")))
    plt.setp(ax.get_xticklabels(), fontsize=7, color=TEXT_PRIMARY, fontfamily="serif",
             rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=8, colors=TEXT_PRIMARY)
    ax.set_title("EUR, deposits excluded, start = 100; the S&P 500 line excludes dividends", fontsize=8,
                 color=TEXT_SECONDARY, fontfamily="serif", pad=6)
    ax.set_ylabel("Indexed (start = 100)", fontsize=8, color=TEXT_SECONDARY,
                  fontfamily="serif")
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT_PRIMARY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(TEXT_SECONDARY)
    ax.set_facecolor(BG_CHART)
    fig.patch.set_facecolor(BG_CHART)
    return _save_fig(fig, "history_line")


def sector_exposure(positions: list[dict]) -> dict[str, float]:
    """EUR value per sector. ETFs are spread across their underlying sectors via sector_weights."""
    sec: dict[str, float] = {}
    for p in positions:
        weights = p.get("sector_weights")
        if weights:
            for sector_name, weight in weights.items():
                sec[sector_name] = sec.get(sector_name, 0) + p["value_eur"] * weight
        else:
            s = p.get("sector") or "Other"
            sec[s] = sec.get(s, 0) + p["value_eur"]
    return sec


def region_exposure(positions: list[dict]) -> dict[str, float]:
    """EUR value per region bucket."""
    geo: dict[str, float] = {}
    for p in positions:
        region = p.get("region") or "Other"
        geo[region] = geo.get(region, 0) + p["value_eur"]
    return geo


# Rough currency mix behind each region bucket (index weights, rounded). Buckets
# group currencies that move together against the euro.
_CCY_GROUP = {"EUR": "EUR", "USD": "USD", "JPY": "JPY", "HKD": "CNY / HKD", "CNY": "CNY / HKD",
              "GBP": "GBP / CHF / Nordic", "CHF": "GBP / CHF / Nordic", "DKK": "GBP / CHF / Nordic",
              "SEK": "GBP / CHF / Nordic", "NOK": "GBP / CHF / Nordic",
              "CAD": "CAD / AUD", "AUD": "CAD / AUD"}
_REGION_CURRENCIES: dict[str, dict[str, float]] = {
    "Global": {"USD": 0.72, "EUR": 0.08, "JPY": 0.055, "GBP / CHF / Nordic": 0.08,
               "CAD / AUD": 0.045, "Other": 0.02},
    "North America": {"USD": 1.0},
    "Europe": {"EUR": 0.55, "GBP / CHF / Nordic": 0.45},
    "Japan": {"JPY": 1.0},
    "China / HK": {"CNY / HKD": 1.0},
    "Emerging Markets": {"CNY / HKD": 0.27, "EM & other Asia": 0.73},
    "Asia Pacific": {"EM & other Asia": 1.0},
    "Latin America": {"EM & other Asia": 1.0},
}


def currency_exposure(positions: list[dict]) -> dict[str, float]:
    """EUR value per underlying currency group, looking through to where the business earns.

    The listing currency says little (a US ETF on Xetra trades in EUR but holds
    dollars), so ETFs and non-European stocks use their region's currency mix.
    European stocks use their listing currency, which tells EUR from CHF or GBP.
    """
    out: dict[str, float] = {}
    for p in positions:
        region = p.get("region") or "Other"
        if region == "Europe" and p.get("sector") != "ETF":
            mix = {_CCY_GROUP.get(p.get("currency", ""), "Other"): 1.0}
        else:
            mix = _REGION_CURRENCIES.get(region, {"Other": 1.0})
        for ccy, w in mix.items():
            out[ccy] = out.get(ccy, 0) + p["value_eur"] * w
    return out


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
    labels, sizes = _prepare_pie(list(sector_exposure(positions).items()), max_slices=8)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    _pie(ax, sizes, labels)
    charts["sector_pie"] = _save_fig(fig, "sector_pie")

    # 3. Geographic allocation — top 8 + Others
    labels, sizes = _prepare_pie(list(region_exposure(positions).items()), max_slices=8)
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
    for bar, val in zip(bars, reversed(vals), strict=True):
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
