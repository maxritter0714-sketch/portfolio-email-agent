"""Claude analysis, headline, actionable parsing, and conviction tracking."""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path

import anthropic

from news import _extract_text_from_response, _first_token, _word_in

log = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
_CONVICTION_MAX_AGE_DAYS = 182  # older recommendations are dropped
_CONVICTION_PATH = Path("conviction.json")


def claude_model() -> str:
    """Model for analysis and headline. Override with CLAUDE_MODEL."""
    return os.getenv("CLAUDE_MODEL", "").strip() or DEFAULT_CLAUDE_MODEL


def _value_vs_jan_line(summary: dict) -> str:
    b = summary.get("value_breakdown")
    if b:
        return (
            f"\nValue change since {b['since'].isoformat()}: €{b['change']:+,.0f} "
            f"(net paid in €{b['invested']:+,.0f}, market gain €{b['market']:+,.0f}; "
            f"dividends/interest to cash €{b['income']:+,.0f})"
        )
    vs_jan = summary.get("value_vs_jan_pct")
    if vs_jan is None:
        return ""
    return (
        f"\nBook value vs January snapshot: {vs_jan:+.2f}% "
        "(includes deposits/withdrawals; not a time-weighted return)"
    )


# ── Claude analysis ───────────────────────────────────────────────────────────
def _build_user_context(positions, summary, benchmarks, port_news, gen_news, metrics, conviction: dict | None = None, momentum: list[dict] | None = None, lookthrough: list[dict] | None = None, events: list[dict] | None = None, allocation: list[dict] | None = None, ref_prices: dict[str, float] | None = None) -> str:
    positions_text = "\n".join(
        f"- {p['name']} ({p['ticker']}): {p['shares']:.4g} shares | "
        f"current €{p['price_eur']:.2f} | "
        f"value €{p['value_eur']:,.2f} | P&L €{p['pl_eur']:+,.2f} ({p['pl_pct']:+.2f}%) | "
        f"1W {p['weekly_pct']:+.2f}% 2W {p['fortnight_pct']:+.2f}% 1M {p['monthly_pct']:+.2f}% YTD {p['ytd_pct']:+.2f}%"
        for p in positions
    )
    bench_text = "\n".join(
        f"- {b['name']}: 1W {b['weekly_pct']:+.2f}% 2W {b['fortnight_pct']:+.2f}% 1M {b['monthly_pct']:+.2f}% YTD {b['ytd_pct']:+.2f}%"
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
        beta_30d_str = f"{metrics['beta_30d']:.2f}" if metrics.get("beta_30d") is not None else "n/a"
        beta_90d_str = f"{metrics['beta_90d']:.2f}" if metrics.get("beta_90d") is not None else "n/a"
        metrics_text = (
            f"\nADVANCED METRICS\n"
            f"Sharpe Ratio: {metrics.get('sharpe', 0):.2f} | "
            f"Sortino Ratio: {metrics.get('sortino', 0):.2f}\n"
            f"Beta vs S&P 500 (EUR): {metrics.get('beta', 0):.2f} | "
            f"Alpha (annualized, vs S&P 500 in EUR): {metrics.get('alpha', 0)*100:+.2f}%\n"
            f"Annualized Volatility: {metrics.get('volatility', 0)*100:.1f}% | "
            f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%\n"
            f"Calmar Ratio: {metrics.get('calmar', 0):.2f} | "
            f"VaR 95% (1-day): {metrics.get('var_95', 0)*100:.2f}%\n"
            f"Correlation to S&P 500 (EUR): {metrics.get('correlation', 0):.2f} | "
            f"HHI Concentration: {metrics.get('hhi', 0):.4f}\n"
            f"Above cost (share of names with price above average buy): "
            f"{metrics.get('win_rate', 0):.1f}%\n"
            f"Rolling Sharpe (90d): {metrics.get('sharpe_90d', 0):.2f} | "
            f"Rolling Beta (30d/90d): {beta_30d_str} / {beta_90d_str}\n"
            f"Rolling Volatility (30d, annualized): {metrics.get('vol_30d', 0)*100:.1f}% | "
            f"CVaR 95% (Expected Shortfall, full 1Y period — not rolling): {metrics.get('cvar_95', 0)*100:.2f}%"
        )
    momentum_text = ""
    if momentum:
        def _fmt_mom(d: dict) -> str:
            return (
                f"- {d['name']} ({d['ticker']}): Q{d['quartile']} | "
                f"1M {d['r1m']*100:+.1f}% 3M {d['r3m']*100:+.1f}% "
                f"6M {d['r6m']*100:+.1f}% 12M {d['r12m']*100:+.1f}%"
            )
        top5 = momentum[:5]
        bottom5 = momentum[5:][-5:]
        momentum_text = (
            f"\n\nMOMENTUM (quartile 4 = strongest, 1 = weakest; score weights "
            f"3M/6M/12M returns — 1M is shown for context but excluded from "
            f"the score since short-term moves tend to mean-revert)\n"
            f"Top:\n" + "\n".join(_fmt_mom(d) for d in top5)
        )
        if bottom5:
            momentum_text += "\nBottom:\n" + "\n".join(_fmt_mom(d) for d in bottom5)
    lookthrough_text = ""
    if lookthrough:
        amplified = [d for d in lookthrough if d["kind"] == "amplified"]
        hidden = [d for d in lookthrough if d["kind"] == "hidden"]
        lookthrough_text = (
            f"\n\nLOOK-THROUGH EXPOSURE (factual only, not a risk signal — more or "
            f"hidden exposure to a name is not inherently good or bad, that depends "
            f"on the reader's own view of it; ETF top_holdings only covers the "
            f"largest ~10 constituents per fund, so this is a conservative floor, "
            f"not exhaustive)\n"
        )
        if amplified:
            lookthrough_text += (
                "Directly held, true exposure amplified ≥20% by ETFs:\n"
                + "\n".join(
                    f"- {d['name']} ({d['ticker']}): direct €{d['direct_eur']:,.0f} | "
                    f"effective €{d['effective_eur']:,.0f} "
                    f"(+{d['hidden_eur']/d['direct_eur']*100:.0f}% via ETFs)"
                    for d in amplified[:6]
                )
            )
        if hidden:
            if amplified:
                lookthrough_text += "\n"
            lookthrough_text += (
                "Not held directly, meaningful exposure via ETFs only:\n"
                + "\n".join(
                    f"- {d['name']} ({d['ticker']}): €{d['effective_eur']:,.0f} "
                    f"({d['hidden_pct']:.2f}% of portfolio)"
                    for d in hidden[:6]
                )
            )
    return (
        f"PORTFOLIO SUMMARY\n"
        f"Total value: €{summary['total_value_eur']:,.2f} | "
        f"Total P&L: €{summary['total_pl_eur']:+,.2f} ({summary['total_pl_pct']:+.2f}%)\n"
        f"Weekly: {summary['weekly_pct']:+.2f}% | Fortnight: {summary['fortnight_pct']:+.2f}% | "
        f"Monthly: {summary['monthly_pct']:+.2f}% | "
        f"YTD (price, current holdings): {summary['ytd_pct']:+.2f}%"
        f"{_value_vs_jan_line(summary)}\n"
        f"Trailing 12M Dividend Income: €{summary.get('total_dividends_ttm_eur', 0):,.2f} "
        f"({summary.get('portfolio_yield_pct', 0):.2f}% yield)\n\n"
        f"POSITIONS\n{positions_text}\n\n"
        f"BENCHMARKS\n{bench_text}"
        f"{metrics_text}"
        f"{momentum_text}"
        f"{lookthrough_text}"
        f"{_build_events_context(events)}"
        f"{_build_allocation_context(allocation)}\n\n"
        f"PORTFOLIO NEWS\n{port_news_text}\n\n"
        f"GENERAL NEWS\n{gen_news_text}"
        + _build_conviction_context(conviction, positions, ref_prices)
    )


def _build_events_context(events: list[dict] | None) -> str:
    if not events:
        return ""
    lines = [
        f"- {e['date'].isoformat()}: {e['kind']} — {e['name']} ({e['ticker']})"
        for e in events
    ]
    return "\n\nUPCOMING EVENTS (next 14 days)\n" + "\n".join(lines)


def _build_allocation_context(allocation: list[dict] | None) -> str:
    if not allocation:
        return ""
    lines = []
    for r in allocation:
        if r["target_pct"] is None:
            lines.append(f"- {r['name']}: {r['actual_pct']:.1f}% (no target)")
        else:
            flag = " — OUTSIDE TOLERANCE" if r["off_target"] else ""
            lines.append(
                f"- {r['name']}: {r['actual_pct']:.1f}% vs target {r['target_pct']:.1f}% "
                f"({r['drift_pct']:+.1f} pts){flag}"
            )
    return "\n\nTARGET ALLOCATION (owner-defined)\n" + "\n".join(lines)


def _build_conviction_context(
    conviction: dict | None, positions: list[dict], ref_prices: dict[str, float] | None = None,
) -> str:
    if not conviction:
        return ""
    price_map = {p["ticker"]: p["price_eur"] for p in positions}
    ref_prices = ref_prices or {}
    lines = []
    for ticker, rec in conviction.items():
        price_at = ref_prices.get(ticker) or rec.get("price_eur", 0.0)
        current = price_map.get(ticker)
        if current and price_at:
            chg = (current - price_at) / price_at * 100
            chg_str = f"{chg:+.1f}% since recommendation"
        else:
            chg_str = "no longer in portfolio"
        lines.append(
            f"- {ticker}: {rec['action']} (Report #{rec['report']}, {rec['date']}, "
            f"€{price_at:.2f} at time) — {chg_str}"
        )
    return "\n\nPRIOR RECOMMENDATIONS\n" + "\n".join(lines)


_PROMPT_PERSONA = (
    "You are a personal financial analyst writing for a private wealth briefing. "
    "Your tone is editorial, precise, confident — like the Financial Times or The "
    "Economist. You have access to the user's full portfolio data, news, benchmarks, "
    "advanced quant metrics including Sharpe, Beta, Alpha, Sortino, VaR, Max "
    "Drawdown, rolling 90d Sharpe and 30d/90d Beta, per-holding multi-timeframe "
    "momentum scores (quartile-ranked, 4=strongest), and LOOK-THROUGH EXPOSURE data "
    "showing stocks embedded inside held ETFs beyond any direct position. When present, "
    "you also get UPCOMING EVENTS (earnings and ex-dividend dates in the next 14 days) "
    "and an owner-defined TARGET ALLOCATION; weigh any group outside its tolerance in "
    "the actionable suggestions. Use these metrics to give specific analysis."
)

_PROMPT_STRUCTURE = (
    "Write a structured report with exactly these four sections:\n\n"
    "## (1) Portfolio Performance Summary — key numbers, what's up, what's down. Where "
    "the MOMENTUM data shows a holding's 1-month move diverging meaningfully from its "
    "3M/6M/12M trend, or a clear quartile 1 (weakest) or quartile 4 (strongest) holding, "
    "name it specifically. Where LOOK-THROUGH EXPOSURE shows a directly-held "
    "stock's true effective weight is meaningfully higher than its direct position "
    "alone, or shows non-trivial exposure to a company held only through ETFs, "
    "mention it as a factual data point — not as a problem to fix. Whether more "
    "or hidden exposure to a given name is desirable depends on the reader's own "
    "view of that company, not on the metric itself.\n"
    "## (2) Benchmark Comparison — how the portfolio did vs S&P 500, Nasdaq, and MSCI "
    "World. ALWAYS cite all three benchmarks — do not omit one because it makes the "
    "comparison look worse. ALWAYS compare the same time period: YTD vs YTD, or "
    "fortnight vs fortnight — never mix since-inception portfolio returns against "
    "YTD benchmark returns. Reference Alpha and Beta where relevant. Where rolling "
    "90d Sharpe or 30d/90d Beta diverge meaningfully from the full-period values, "
    "note whether risk-adjusted returns or market sensitivity are improving or "
    "deteriorating.\n"
    "## (3) News & Market Context — the 3 most important implications of the news for "
    "this specific portfolio. Format EACH item on its own line as:\n"
    "[IMPLICATION] Title — One sentence connecting the news to a specific holding or theme.\n"
    "where IMPLICATION is the literal prefix. Do not summarise the news — state what it "
    "means for this portfolio. Do not use bullets or numbers — just the [IMPLICATION] prefix.\n"
    "## (4) Actionable Suggestions — 3 to 5 specific recommendations. Format EACH item "
    "on its own line as:\n"
    "[ACTION] Title — Detail sentence.\n"
    "where ACTION is EXACTLY one of: TRIM, EXIT, RESEARCH, ADD. Do not use bullets or "
    "numbers — just the [ACTION] prefix. Example:\n"
    "[TRIM] Reduce NVDA concentration — At 22% of portfolio, HHI is elevated; "
    "trimming to 15% would still capture upside while diversifying tail risk.\n"
    "CRITICAL: any open recommendation from the PRIOR RECOMMENDATIONS data that is "
    "still relevant MUST appear here as an [ACTION] item — do not let it disappear "
    "from the actionable section simply because it was mentioned in prose in section (3)."
)

_PROMPT_FORMATTING = (
    "FORMATTING RULES: "
    "Never use Markdown table syntax (pipe characters and dashes). "
    "Never use bullet points (- or *) — use plain paragraphs. "
    "When presenting benchmark comparisons or tabular data, write it as flowing prose sentences. "
    "Example: 'The S&P 500 returned +8.35% YTD, Nasdaq +11.42%, and MSCI World +8.48% — "
    "all well below this portfolio\\'s +23.07%.' "
    "Do not use any HTML tags. "
    "All output must be plain text with Markdown bold (**text**) only. "
    "Use **bold** sparingly for headline figures. Be direct and concise. "
    "Never add disclaimers, caveats, or 'this is not financial advice' language. "
    "The reader is the portfolio owner — treat them as an informed adult. "
    "DIVERSIFICATION: the HHI measures position-count concentration across holdings — "
    "it says nothing about factor risk. A portfolio with HHI 0.03 and Beta 1.9 concentrated "
    "in one sector is not 'well diversified' — it has low position concentration but high "
    "factor concentration. Never conflate the two; if Beta > 1.5 or a single sector exceeds "
    "35% of NAV, acknowledge the factor risk even if HHI is low. "
    "STATISTICAL CONFIDENCE: annualised metrics (Alpha, Sharpe, Sortino, Calmar) derived "
    "from less than two years of history have wide confidence intervals — a single bull or "
    "bear phase dominates. Never write that short-history metrics 'confirm', 'prove', or "
    "'formally establish' outperformance. State the figure, note it is based on limited "
    "history, and let the reader draw their own conclusion. "
    "YTD for holdings and the portfolio masthead is a weighted euro price return of "
    "current names this calendar year. Benchmark 1W/2W/1M/YTD figures are also euro "
    "returns (index converted at daily EUR FX), so portfolio and benchmark figures "
    "compare like-for-like. If a book-value-vs-January figure is present, "
    "it includes deposits and withdrawals and is not a time-weighted return; do not "
    "call it YTD performance."
)


def _assert_analysis_complete(text: str) -> None:
    """Raise if the analysis text appears truncated (last line doesn't end a sentence)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise RuntimeError("Analysis is empty")
    last = lines[-1]
    if not re.search(r'[.!?"]$', last) and not last.startswith("[ACTION]") and not last.startswith("[IMPLICATION]"):
        raise RuntimeError(
            f"Analysis appears truncated — last line: {last[:80]!r}. "
            "Aborting to avoid sending a broken email."
        )


def call_claude(positions, summary, benchmarks, port_news, gen_news, metrics, conviction: dict | None = None, momentum: list[dict] | None = None, lookthrough: list[dict] | None = None, events: list[dict] | None = None, allocation: list[dict] | None = None, ref_prices: dict[str, float] | None = None) -> str:
    """Return Claude's analysis as markdown (sections 1-4)."""
    client = anthropic.Anthropic()

    system_prompt = "\n\n".join([_PROMPT_PERSONA, _PROMPT_STRUCTURE, _PROMPT_FORMATTING])

    user_content = _build_user_context(positions, summary, benchmarks, port_news, gen_news, metrics, conviction, momentum, lookthrough, events, allocation, ref_prices)

    log.info("Calling Claude API for analysis (model: %s)…", claude_model())
    response = client.messages.create(
        model=claude_model(),
        # Headroom for adaptive thinking, which counts toward max_tokens.
        max_tokens=16000,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_content}],
    )
    log.info(
        "Claude analysis received (input tokens: %d, cached: %d, stop_reason: %s)",
        response.usage.input_tokens,
        response.usage.cache_read_input_tokens or 0,
        response.stop_reason,
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            "Analysis truncated (stop_reason=max_tokens). "
            "Aborting to avoid sending a broken email."
        )
    return _extract_text_from_response(response)


def call_claude_headline(summary: dict, benchmarks: list[dict], metrics: dict) -> str:
    """Return a single-sentence editorial headline (<=20 words, one number)."""
    client = anthropic.Anthropic()

    system = (
        "You write a single-sentence editorial headline for a fortnightly portfolio "
        "briefing, styled like the Financial Times. Requirements: max 20 words; MUST "
        "contain at least one specific number, percentage, or comparison; subtly "
        "insightful; no filler. Output ONLY the sentence — no quotes, no preamble. "
        "CRITICAL: use the most recent and most adverse move as your frame — do NOT "
        "choose the less alarming time window to soften a bad result. If the weekly "
        "move is worse than the fortnight, lead with the weekly figure. Use price YTD "
        "for performance claims, not book value vs January."
    )

    bench_lines = "\n".join(
        f"{b['name']}: fortnight {b['fortnight_pct']:+.2f}%, monthly {b['monthly_pct']:+.2f}%, YTD {b['ytd_pct']:+.2f}%"
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
        f"Portfolio weekly: {summary['weekly_pct']:+.2f}%\n"
        f"Portfolio fortnight: {summary['fortnight_pct']:+.2f}%\n"
        f"Portfolio monthly: {summary['monthly_pct']:+.2f}%\n"
        f"Portfolio YTD (price, current holdings): {summary['ytd_pct']:+.2f}%"
        f"{_value_vs_jan_line(summary)}\n"
        f"Total P&L: €{summary['total_pl_eur']:+,.0f} ({summary['total_pl_pct']:+.2f}%)\n"
        f"{bench_lines}\n"
        f"{metrics_line}"
    )

    log.info("Calling Claude API for headline…")
    response = client.messages.create(
        model=claude_model(),
        # One sentence: keep thinking light, but leave room for it in max_tokens.
        max_tokens=2000,
        output_config={"effort": "low"},
        system=[{"type": "text", "text": system}],
        messages=[{"role": "user", "content": context}],
    )
    text = _extract_text_from_response(response).strip()
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


def load_conviction() -> dict:
    """Load persisted per-ticker recommendations from conviction.json."""
    if _CONVICTION_PATH.exists():
        try:
            return json.loads(_CONVICTION_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _unique_ticker(hits: list[str]) -> str | None:
    uniq = list(dict.fromkeys(hits))
    if len(uniq) == 1:
        return uniq[0]
    return None


def match_action_ticker(item: dict, positions: list[dict]) -> str | None:
    """Ticker for an actionable line: symbol first, then company name.

    Name matching uses the first significant word (e.g. Nvidia → NVDA).
    Ambiguous hits (two holdings in one line) return None.
    """
    title = item.get("title", "") or ""
    detail = item.get("detail", "") or ""
    combined = f"{title} {detail}".strip()

    ticker_hits: list[str] = []
    for haystack in (title, combined):
        ticker_hits = [p["ticker"] for p in positions if _word_in(haystack, p["ticker"])]
        if ticker_hits:
            break

    name_hits: list[str] = []
    for haystack in (title, combined):
        name_hits = []
        for p in positions:
            token = _first_token(p["name"])
            if len(token) < 4:
                continue
            if _word_in(haystack, token):
                name_hits.append(p["ticker"])
        if name_hits:
            break

    return _unique_ticker(ticker_hits + name_hits)


def save_conviction(
    conviction: dict,
    actionable_items: list[dict],
    positions: list[dict],
    issue_num: int,
) -> None:
    """Update conviction.json with any tickers mentioned in this run's actions."""
    price_map = {p["ticker"]: p["price_eur"] for p in positions}
    today = date.today().isoformat()
    cutoff = (date.today() - timedelta(days=_CONVICTION_MAX_AGE_DAYS)).isoformat()
    for ticker in [t for t, rec in conviction.items() if rec.get("date", "") < cutoff]:
        del conviction[ticker]
    for item in actionable_items:
        matched = match_action_ticker(item, positions)
        if matched:
            conviction[matched] = {
                "action": item["action"],
                "report": issue_num,
                "date": today,
                "price_eur": price_map.get(matched, 0.0),
            }
    _CONVICTION_PATH.write_text(json.dumps(conviction, indent=2), encoding="utf-8")


# ── Recommendation track record ───────────────────────────────────────────────
_HIT_DIRECTION = {"ADD": 1, "TRIM": -1, "EXIT": -1}


def conviction_scorecard(
    conviction: dict, positions: list[dict], ref_prices: dict[str, float] | None = None,
) -> list[dict]:
    """How each prior recommendation's price moved since it was made.

    ADD is a hit when the price rose; TRIM and EXIT when it fell. RESEARCH has
    no direction. Positions no longer held have no current price. `ref_prices`
    (split-adjusted) take precedence over the price stored at the time.
    """
    price_map = {p["ticker"]: p["price_eur"] for p in positions}
    ref_prices = ref_prices or {}
    rows: list[dict] = []
    for ticker, rec in conviction.items():
        price_then = ref_prices.get(ticker) or rec.get("price_eur") or 0.0
        price_now = price_map.get(ticker)
        change_pct = None
        if price_now and price_then:
            change_pct = (price_now - price_then) / price_then * 100
        direction = _HIT_DIRECTION.get(rec.get("action", ""))
        hit = None
        if direction is not None and change_pct:
            hit = (change_pct > 0) == (direction > 0)
        rows.append(dict(
            ticker=ticker, action=rec.get("action", ""), report=rec.get("report", 0),
            date=rec.get("date", ""), price_then=price_then, price_now=price_now,
            change_pct=change_pct, hit=hit,
        ))
    rows.sort(key=lambda r: (-r["report"], r["ticker"]))
    return rows
