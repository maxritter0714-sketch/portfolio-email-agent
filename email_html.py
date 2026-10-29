"""HTML email builder and metric interpretation copy."""
from __future__ import annotations

import html
import os
import re
from datetime import date

from theme import (
    BG_CREAM, BURGUNDY, CHARCOAL, DARK_GOLD, GOLD, NAVY,
    RULE_FAINT, SAGE, TEXT_PRIMARY, TEXT_SECONDARY,
)

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


def interp_yield(v):
    if v < 0.5: return "Growth-oriented, minimal income"
    if v < 1.5: return "Modest income contribution"
    if v < 3.0: return "Meaningful income contribution"
    return "Income-focused profile"


# ── HTML helpers ──────────────────────────────────────────────────────────────
def _color(val: float) -> str:
    return SAGE if val >= 0 else BURGUNDY


def _fmt(val: float) -> str:
    return f"{val:+.2f}%"


def _eur_signed(val: float, decimals: int = 0) -> str:
    """Format a signed euro amount with sign before the symbol: -€254, +€254."""
    sign = "-" if val < 0 else "+"
    return f"{sign}€{abs(val):,.{decimals}f}"


def _pl_from_pct(value: float, pct: float) -> float:
    """Euro change behind a return that ended at `value`."""
    denom = 100.0 + pct
    return value * pct / denom if denom else 0.0


def _mast_stat_cell(
    label: str, pct: float, width: str, caption: str = "", amount: float | None = None,
) -> str:
    extra = ""
    if amount is not None:
        extra += (
            f'<div style="font-family:Georgia,serif;font-size:12px;'
            f'color:{TEXT_SECONDARY};margin-top:4px;font-variant-numeric:tabular-nums">'
            f"{_eur_signed(amount)}</div>"
        )
    if caption:
        extra += (
            f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:8px;'
            f'color:{TEXT_SECONDARY};margin-top:4px;line-height:1.3">{caption}</div>'
        )
    return (
        f'<td style="padding-top:16px;width:{width};vertical-align:top">'
        f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:9px;'
        f'text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};'
        f'margin-bottom:6px">{label}</div>'
        f'<div style="font-family:Georgia,serif;font-size:20px;color:{_color(pct)};'
        f'font-variant-numeric:tabular-nums">{pct:+.2f}%</div>'
        f"{extra}</td>"
    )


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


def _analysis_with_drop_cap(analysis_html: str) -> str:
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

    return re.sub(r"(<p[^>]*>)(.*?)</p>", repl, analysis_html, count=1, flags=re.DOTALL)


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


_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def _small_td(content: str, align: str = "left", color: str = TEXT_PRIMARY) -> str:
    return (
        f'<td style="padding:10px 4px;border-bottom:1px solid {RULE_FAINT};'
        f'font-family:Arial,Helvetica,sans-serif;font-size:13px;text-align:{align};'
        f'font-variant-numeric:tabular-nums;color:{color}">{content}</td>'
    )


def _small_th(label: str, align: str = "left") -> str:
    return (
        f'<th style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
        f'text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};'
        f'text-align:{align};padding:8px 4px;border-bottom:1px solid {TEXT_PRIMARY};'
        f'font-weight:600">{label}</th>'
    )


def _allocation_html(allocation: list[dict]) -> str:
    rows = ""
    for r in allocation:
        target = f'{r["target_pct"]:.1f}%' if r["target_pct"] is not None else "&ndash;"
        if r["drift_pct"] is None:
            drift, drift_color = "&ndash;", TEXT_SECONDARY
        else:
            drift = f'{r["drift_pct"]:+.1f} pts'
            drift_color = BURGUNDY if r["off_target"] else TEXT_PRIMARY
        rows += (
            "<tr>"
            + _small_td(html.escape(r["name"]))
            + _small_td(f'{r["actual_pct"]:.1f}%', "right")
            + _small_td(target, "right")
            + _small_td(drift, "right", drift_color)
            + "</tr>"
        )
    return (
        '<table width="100%" cellpadding="0" cellspacing="0"><thead><tr>'
        + _small_th("Group") + _small_th("Actual", "right")
        + _small_th("Target", "right") + _small_th("Drift", "right")
        + f"</tr></thead><tbody>{rows}</tbody></table>"
    )


_MAX_EVENTS_SHOWN = 12


def _events_html(events: list[dict]) -> str:
    """Earliest events as a table; the rest collapse into a '+ N more' line."""
    shown, hidden = events[:_MAX_EVENTS_SHOWN], len(events) - _MAX_EVENTS_SHOWN
    rows = ""
    for e in shown:
        rows += (
            "<tr>"
            + _small_td(e["date"].strftime("%a %d %b"))
            + _small_td(html.escape(e["kind"]))
            + _small_td(
                f'{html.escape(e["name"])} <span style="font-size:10px;color:{TEXT_SECONDARY};'
                f'letter-spacing:1px">{html.escape(e["ticker"])}</span>'
            )
            + "</tr>"
        )
    more = ""
    if hidden > 0:
        more = (
            f'<p style="font-family:Arial,Helvetica,sans-serif;font-size:11px;'
            f'color:{TEXT_SECONDARY};margin:12px 0 0">'
            f'+ {hidden} more in the next 14 days</p>'
        )
    return (
        '<table width="100%" cellpadding="0" cellspacing="0"><thead><tr>'
        + _small_th("Date") + _small_th("Event") + _small_th("Holding")
        + f"</tr></thead><tbody>{rows}</tbody></table>{more}"
    )


def _scorecard_html(scorecard: list[dict]) -> str:
    calls = [r for r in scorecard if r["hit"] is not None]
    hits = sum(1 for r in calls if r["hit"])
    summary = ""
    if calls:
        summary = (
            f'<p style="font-family:Georgia,serif;font-size:15px;margin:0 0 16px">'
            f'{hits} of {len(calls)} directional calls moved the right way so far.</p>'
        )
    rows = ""
    for r in scorecard:
        if r["change_pct"] is None:
            change, change_color = "no longer held", TEXT_SECONDARY
        else:
            change, change_color = _fmt(r["change_pct"]), _color(r["change_pct"])
        if r["hit"] is None:
            verdict = "&ndash;"
        else:
            verdict = "&#10003;" if r["hit"] else "&#10007;"
        rows += (
            "<tr>"
            + _small_td(html.escape(r["ticker"]))
            + _small_td(html.escape(r["action"]))
            + _small_td(f'No. {r["report"]}', "right", TEXT_SECONDARY)
            + _small_td(change, "right", change_color)
            + _small_td(verdict, "center")
            + "</tr>"
        )
    return (
        summary
        + '<table width="100%" cellpadding="0" cellspacing="0"><thead><tr>'
        + _small_th("Ticker") + _small_th("Call") + _small_th("Issue", "right")
        + _small_th("Since", "right") + _small_th("Hit", "center")
        + f"</tr></thead><tbody>{rows}</tbody></table>"
    )


HOLDINGS_STALE_DAYS = 30


def _breakdown_html(b: dict) -> str:
    """Masthead line: value change since the baseline, split into paid in vs market."""
    def part(label: str, val: float) -> str:
        return (
            f'<span style="color:{TEXT_SECONDARY}">{label}</span> '
            f'<span style="color:{_color(val)}">{_eur_signed(val)}</span>'
        )
    return (
        f'<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:18px">'
        f'<tr><td style="font-family:Arial,Helvetica,sans-serif;font-size:12px;line-height:1.7;'
        f'color:{BG_CREAM}">'
        f'<div style="font-size:9px;text-transform:uppercase;letter-spacing:2px;'
        f'color:{TEXT_SECONDARY};margin-bottom:4px">Since {b["since"].strftime("%d %b %Y")}</div>'
        f'{part("Value", b["change"])} &nbsp;=&nbsp; {part("paid in", b["invested"])} '
        f'&nbsp;+&nbsp; {part("market", b["market"])}'
        f' &nbsp;&middot;&nbsp; {part("dividends &amp; interest", b["income"])}'
        f'</td></tr></table>'
    )


def _data_warning_html(
    today: date, holdings_as_of: date | None, unmapped: list[dict],
    flows_as_of: date | None = None,
) -> str:
    """Banner under the masthead when the holdings data can't be fully trusted."""
    lines = []
    if flows_as_of and (today - flows_as_of).days > HOLDINGS_STALE_DAYS:
        lines.append(
            f"Cash-flow data ends {flows_as_of.strftime('%d %b %Y')}, so the paid-in vs "
            f"market split is unreliable — refresh the Finanzfluss transactions export."
        )
    if holdings_as_of and (today - holdings_as_of).days > HOLDINGS_STALE_DAYS:
        lines.append(
            f"Holdings data is {(today - holdings_as_of).days} days old "
            f"(exported {holdings_as_of.strftime('%d %b %Y')}). Purchases and sales "
            f"since then are missing — refresh the Finanzfluss export."
        )
    if unmapped:
        n = len(unmapped)
        lines.append(
            f"{n} holding{'s' if n != 1 else ''} could not be mapped to a ticker and "
            f"{'are' if n != 1 else 'is'} missing from this report — see the notes at the end."
        )
    if not lines:
        return ""
    return (
        f'<div class="sect-pad" style="padding:20px 40px 0">'
        f'<div style="border-left:3px solid {BURGUNDY};padding:10px 14px;'
        f'font-family:Arial,Helvetica,sans-serif;font-size:12px;line-height:1.5;'
        f'color:{BURGUNDY}">{"<br>".join(lines)}</div></div>'
    )


def _optional_sections_html(
    allocation: list[dict], events: list[dict], scorecard: list[dict],
) -> str:
    """Sections VII+ — each shown only when it has content."""
    sections = []
    if allocation:
        sections.append(("Allocation vs Target", _allocation_html(allocation)))
    if events:
        sections.append(("Coming Up", _events_html(events)))
    if scorecard:
        sections.append(("Track Record", _scorecard_html(scorecard)))
    out = ""
    for i, (label, body) in enumerate(sections, start=6):
        out += (
            f'<div class="sect-pad" style="padding:24px 40px 40px">'
            f"{_sect_header(_ROMAN[i], label)}{body}</div>\n"
        )
    return out


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
    unknown_fx: list[dict] | None = None,
    skipped: list[dict] | None = None,
    allocation: list[dict] | None = None,
    events: list[dict] | None = None,
    scorecard: list[dict] | None = None,
    holdings_as_of: date | None = None,
    unmapped: list[dict] | None = None,
) -> str:
    today = date.today()
    breakdown = summary.get("value_breakdown")
    warning_html = _data_warning_html(
        today, holdings_as_of, unmapped or [],
        breakdown["flows_as_of"] if breakdown else None,
    )
    breakdown_html = _breakdown_html(breakdown) if breakdown else ""
    today_str = today.strftime("%B %-d, %Y") if os.name != "nt" else today.strftime("%B %d, %Y").replace(" 0", " ")

    total = summary["total_value_eur"]
    fortnight_pct = summary["fortnight_pct"]
    fortnight_pl = _pl_from_pct(total, fortnight_pct)
    fortnight_color = _color(fortnight_pl)
    vs_jan       = summary.get("value_vs_jan_pct")
    lifetime     = summary.get("lifetime")
    # (label, pct, euro amount, caption); fortnight already leads the masthead.
    stats = [
        (label, summary[key], _pl_from_pct(total, summary[key]), "")
        for label, key in (("Weekly", "weekly_pct"), ("Monthly", "monthly_pct"), ("YTD", "ytd_pct"))
    ]
    if lifetime:
        stats.append(("Total", lifetime["pct"], lifetime["gain"], ""))
    if vs_jan is not None:
        stats.append(("Vs January", vs_jan, None, "Includes cash in/out"))
    mast_w = f"{100 // len(stats)}%"
    mast_stats = "".join(
        _mast_stat_cell(label, pct, mast_w, caption, amount)
        for label, pct, amount, caption in stats
    )

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
            f'<td class="metric-p-td" style="text-align:center;padding:22px 10px;{left_border}width:25%;vertical-align:top">'
            f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};'
            f'margin-bottom:10px">{label}</div>'
            f'<div class="metric-p-val" style="font-family:Georgia,serif;font-size:26px;color:{TEXT_PRIMARY};'
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
        ("Above cost", f"{metrics.get('win_rate', 0):.0f}%",     interp_wr(metrics.get("win_rate", 0))),
        ("Div Yield (TTM)", f"{summary.get('portfolio_yield_pct', 0):.2f}%", interp_yield(summary.get("portfolio_yield_pct", 0))),
    ]
    # Pad to multiple of 4
    while len(secondary) % 4 != 0:
        secondary.append(("", "", ""))

    sec_rows = ""
    for i in range(0, len(secondary), 4):
        sec_rows += '<tr class="metric-s-row">'
        for j in range(4):
            label, value, interp = secondary[i + j]
            if not label:
                sec_rows += '<td class="metric-s-td" style="width:25%"></td>'
                continue
            sec_rows += (
                f'<td class="metric-s-td" style="padding:16px 12px;border-bottom:1px solid {RULE_FAINT};'
                f'vertical-align:top;width:25%">'
                f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:9px;'
                f'text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};'
                f'margin-bottom:4px">{label}</div>'
                f'<div class="metric-s-val" style="font-family:Georgia,serif;font-size:18px;color:{TEXT_PRIMARY};'
                f'font-variant-numeric:tabular-nums">{value}</div>'
                f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
                f'color:{TEXT_SECONDARY};margin-top:3px;line-height:1.35">{interp}</div>'
                f'</td>'
            )
        sec_rows += "</tr>"

    # ── Positions table ───────────────────────────────────────────────────────
    def _th(label: str, align: str = "left", cls: str = "") -> str:
        cls_attr = f' class="{cls}"' if cls else ""
        return (
            f'<th{cls_attr} style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};'
            f'text-align:{align};padding:10px 4px;border-bottom:1px solid {TEXT_PRIMARY};'
            f'font-weight:600">{label}</th>'
        )

    pos_rows_html = ""
    for p in positions:
        pos_rows_html += (
            f'<tr>'
            f'<td class="pos-name-cell" style="padding:12px 4px;border-bottom:1px solid {RULE_FAINT};'
            f'font-family:Georgia,serif;font-size:14px">'
            f'{html.escape(p["name"])}'
            f' <span style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'color:{TEXT_SECONDARY};letter-spacing:1px;margin-left:6px">{html.escape(p["ticker"])}</span>'
            f'</td>'
            f'<td class="hide-mobile" style="padding:12px 4px;border-bottom:1px solid {RULE_FAINT};'
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
        ("history_line",  "Portfolio vs S&P 500 (Indexed)"),
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
            f'<a href="{html.escape(a["url"])}" style="text-decoration:none;color:{TEXT_PRIMARY}">'
            f'<div style="font-family:Georgia,serif;font-size:15px;line-height:1.45;'
            f'margin-bottom:8px">{html.escape(a["headline"])}</div>'
            f'</a>'
            f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'color:{TEXT_SECONDARY};text-transform:uppercase;letter-spacing:1.5px">'
            f'{html.escape(a["source"])} &middot; {html.escape(a["date"])}{tickers_html}'
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
            f'<a href="{html.escape(a["url"])}" style="text-decoration:none;color:{TEXT_PRIMARY}">'
            f'<div style="font-family:Georgia,serif;font-size:15px;line-height:1.45;'
            f'margin-bottom:8px">{html.escape(a["headline"])}</div>'
            f'</a>'
            f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;'
            f'color:{TEXT_SECONDARY};text-transform:uppercase;letter-spacing:1.5px">'
            f'{html.escape(a["source"])} &middot; {html.escape(a["date"])}'
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
        title = html.escape(item.get("title", ""))
        detail = html.escape(item.get("detail", ""))
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

    unknown_fx = unknown_fx or []
    skipped = skipped or []
    unmapped = unmapped or []
    notes_parts: list[str] = []
    if holdings_as_of:
        notes_parts.append(f"Holdings as of {holdings_as_of.strftime('%d %b %Y')}.")
    if breakdown:
        notes_parts.append(
            f"Cash flows as of {breakdown['flows_as_of'].strftime('%d %b %Y')}; "
            "market = value change minus net money paid in."
        )
    notes_parts.append(
        "Holding and index 1W/2W/1M/YTD figures are euro returns."
    )
    if unmapped:
        missing = ", ".join(
            f"{html.escape(u.get('name') or '?')} ({html.escape(u.get('isin') or '?')})"
            for u in unmapped
        )
        notes_parts.append(f"Not in this report (ISIN not mapped to a ticker): {missing}.")
    if metrics:
        notes_parts.append(
            "Beta, alpha, and correlation are versus the S&P 500 in euro terms."
        )
    if skipped:
        omitted = ", ".join(
            html.escape(s.get("ticker") or s.get("name") or "?") for s in skipped
        )
        notes_parts.append(f"Omitted (no price data): {omitted}.")
    if unknown_fx:
        fx_note = ", ".join(
            f"{html.escape(s.get('ticker', '?'))} ({html.escape(s.get('currency', '?'))})"
            for s in unknown_fx
        )
        notes_parts.append(f"Unknown FX converted via USD fallback: {fx_note}.")
    data_notes_html = ""
    if notes_parts:
        data_notes_html = (
            f'<div class="sect-pad" style="padding:0 40px 24px">'
            f'<p style="font-family:Arial,Helvetica,sans-serif;font-size:11px;'
            f'color:{TEXT_SECONDARY};line-height:1.55;margin:0">'
            f'{" ".join(notes_parts)}'
            f'</p></div>'
        )

    # ── Full HTML ─────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Fortnight &middot; No. {issue_num}</title>
<style>
@media only screen and (max-width:480px){{
  .mast-pad{{padding:20px 16px!important}}
  .mast-h1{{font-size:26px!important;margin:14px 0 4px!important}}
  .mast-num{{font-size:22px!important}}
  .mast-num-sub{{font-size:13px!important}}
  .mast-col{{display:block!important;width:100%!important;padding-right:0!important;margin-bottom:10px}}
  .sect-pad{{padding:14px 16px!important}}
  .sect-pad-top{{padding:24px 16px 14px!important}}
  .sect-pad-metrics{{padding:8px 16px!important}}
  .sect-pad-sec{{padding:4px 16px 14px!important}}
  .headline-p{{font-size:17px!important}}
  .metric-p-table{{display:block!important;width:100%!important}}
  .metric-p-row{{display:flex!important;flex-wrap:wrap!important}}
  .metric-p-td{{width:50%!important;box-sizing:border-box!important;border-left:none!important;border-bottom:1px solid #e8e3d8!important}}
  .metric-p-val{{font-size:20px!important}}
  .metric-s-table{{display:block!important;width:100%!important}}
  .metric-s-row{{display:flex!important;flex-wrap:wrap!important}}
  .metric-s-td{{width:50%!important;box-sizing:border-box!important}}
  .metric-s-val{{font-size:15px!important}}
  .hide-mobile{{display:none!important}}
  .pos-name-cell{{font-size:12px!important}}
}}
</style>
</head>
<body style="margin:0;padding:0;background:{BG_CREAM};font-family:Georgia,serif;color:{TEXT_PRIMARY}">
<div style="max-width:640px;margin:0 auto;background:{BG_CREAM}">

<!-- Masthead -->
<table width="100%" cellpadding="0" cellspacing="0" style="background:{CHARCOAL};border-bottom:3px solid {GOLD}">
<tr><td class="mast-pad" style="padding:40px">
  <table width="100%" cellpadding="0" cellspacing="0"><tr>
    <td style="font-family:Arial,Helvetica,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY}">Portfolio Report &middot; No. {issue_num}</td>
    <td style="font-family:Arial,Helvetica,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};text-align:right">{today_str}</td>
  </tr></table>
  <h1 class="mast-h1" style="font-family:Georgia,serif;font-size:38px;font-weight:normal;margin:28px 0 6px;color:{BG_CREAM};letter-spacing:-0.5px">The Fortnight</h1>
  <p style="font-family:Georgia,serif;font-style:italic;color:{GOLD};margin:0 0 36px;font-size:14px">A private wealth briefing</p>
  <table width="100%" cellpadding="0" cellspacing="0"><tr>
    <td class="mast-col" style="padding-right:32px;width:50%;vertical-align:top">
      <div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};margin-bottom:8px">Total Value</div>
      <div class="mast-num" style="font-family:Georgia,serif;font-size:32px;color:{BG_CREAM};font-variant-numeric:tabular-nums">€{total:,.0f}</div>
    </td>
    <td class="mast-col" style="width:50%;vertical-align:top">
      <div style="font-family:Arial,Helvetica,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY};margin-bottom:8px">Fortnight P&amp;L</div>
      <div class="mast-num" style="font-family:Georgia,serif;font-size:32px;color:{fortnight_color};font-variant-numeric:tabular-nums">{_eur_signed(fortnight_pl)} <span class="mast-num-sub" style="font-size:18px;color:{TEXT_SECONDARY}">({fortnight_pct:+.2f}%)</span></div>
    </td>
  </tr></table>
  <table width="100%" cellpadding="0" cellspacing="0" style="margin-top:24px;border-top:1px solid rgba(255,255,255,0.15)">
    <tr>
      {mast_stats}
    </tr>
  </table>
  {breakdown_html}
</td></tr>
</table>

{warning_html}
<!-- The Headline -->
<div class="sect-pad-top" style="padding:48px 40px 24px">
  <p class="headline-p" style="font-family:Georgia,serif;font-size:22px;line-height:1.45;color:{TEXT_PRIMARY};margin:0">{headline_html}</p>
</div>

<!-- Primary metric strip -->
<div class="sect-pad-metrics" style="padding:8px 40px">
  <table class="metric-p-table" width="100%" cellpadding="0" cellspacing="0" style="border-top:1px solid {TEXT_PRIMARY};border-bottom:1px solid {TEXT_PRIMARY}">
    <tr class="metric-p-row">{primary_cells}</tr>
  </table>
</div>

<!-- Secondary metric strip -->
<div class="sect-pad-sec" style="padding:4px 40px 24px">
  <table class="metric-s-table" width="100%" cellpadding="0" cellspacing="0">{sec_rows}</table>
</div>

<!-- Section I. Positions -->
<div class="sect-pad" style="padding:24px 40px">
  {_sect_header("I", "Positions")}
  <table width="100%" cellpadding="0" cellspacing="0">
    <thead><tr>
      {_th("Holding", "left")}
      {_th("Shares", "right", "hide-mobile")}
      {_th("Value", "right")}
      {_th("P&amp;L", "right")}
      {_th("YTD", "right")}
    </tr></thead>
    <tbody>{pos_rows_html}</tbody>
  </table>
</div>

<!-- Section II. By the Numbers -->
<div class="sect-pad" style="padding:24px 40px">
  {_sect_header("II", "By the Numbers")}
  {figs_html}
</div>

<!-- Section III. The Analysis -->
<div class="sect-pad" style="padding:24px 40px">
  {_sect_header("III", "The Analysis")}
  <div style="font-family:Georgia,serif;font-size:15px;line-height:1.7;color:{TEXT_PRIMARY}">
    {analysis_html}
  </div>
</div>

<!-- Section IV. Portfolio News -->
<div class="sect-pad" style="padding:24px 40px">
  {_sect_header("IV", "Portfolio News")}
  {port_news_html}
</div>

<!-- Section V. The World -->
<div class="sect-pad" style="padding:24px 40px">
  {_sect_header("V", "The World")}
  {gen_news_html}
</div>

<!-- Section VI. Actionable -->
<div class="sect-pad" style="padding:24px 40px 48px">
  {_sect_header("VI", "Actionable")}
  {actionable_html}
</div>

{_optional_sections_html(allocation or [], events or [], scorecard or [])}
{data_notes_html}

<!-- Footer -->
<table width="100%" cellpadding="0" cellspacing="0" style="background:{CHARCOAL};border-top:2px solid {GOLD}">
<tr><td style="padding:32px 40px;text-align:center;font-family:Arial,Helvetica,sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:2px;color:{TEXT_SECONDARY}">
The Fortnight &middot; No. {issue_num} &middot; {today_str}
</td></tr>
</table>

</div>
</body>
</html>"""
