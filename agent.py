#!/usr/bin/env python3
"""
Portfolio Email Agent
Fetches market data, news, and Claude AI analysis, then sends an editorial-style
HTML portfolio report via Gmail. Collects data every Saturday; sends the email
every other Saturday (every 14 days from 11 Jan 2025). Bypass with --force.
"""

import argparse
import logging
import os
import smtplib
import sys
import tempfile
import traceback
from datetime import date
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import yfinance as yf
from dotenv import load_dotenv

from claude_report import (
    _assert_analysis_complete,
    call_claude,
    call_claude_headline,
    conviction_scorecard,
    load_conviction,
    parse_actionable,
    save_conviction,
    split_analysis,
)
from email_html import build_html_email
from market import (
    _hist_cache,
    append_history,
    compute_allocation_drift,
    fetch_benchmarks,
    fetch_portfolio_data,
    fetch_upcoming_events,
    generate_charts,
    get_fx_rates,
    get_history_ytd,
    holdings_as_of,
    is_send_saturday,
    issue_number,
    load_portfolio,
    load_cashflows,
    load_targets,
    load_unmapped_isins,
    next_portfolio_index,
    recommendation_ref_prices,
    value_change_breakdown,
)
from metrics import (
    compute_advanced_metrics,
    compute_lookthrough_exposure,
    compute_momentum_scores,
)
from news import (
    fetch_general_news,
    fetch_portfolio_news,
    filter_news_by_relevance,
)

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


# ── History ───────────────────────────────────────────────────────────────────
def record_history(positions: list[dict], summary: dict, fx_rates: dict[str, float]) -> None:
    """Append today's data point, including the deposit-free portfolio index."""
    source = "scheduled" if os.environ.get("GITHUB_ACTIONS") == "true" else "manual"
    port_index = next_portfolio_index(positions, fx_rates)
    sp500 = _hist_cache.get("^GSPC")
    sp500_eur = None
    if sp500 is not None and not sp500.empty:
        sp500_eur = float(sp500.iloc[-1]) * fx_rates.get("USD", 0.0) or None
    append_history(summary, issue_number(), source=source,
                   port_index=port_index, sp500_eur=sp500_eur)
    log.info("History updated — total value EUR %.0f, index %.2f (source=%s)",
             summary["total_value_eur"], port_index, source)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio Email Agent — sends every other Saturday"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the fortnight schedule check and run immediately",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Build HTML and write to preview.html instead of sending email",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Fetch prices and append to history.csv only — no Claude, no email",
    )
    args = parser.parse_args()

    if args.data_only:
        log.info("=== Portfolio Agent (data-only) ===")
        try:
            rows = load_portfolio("portfolio.csv")
            fx_rates = get_fx_rates()
            positions, summary, unknown_fx, skipped = fetch_portfolio_data(rows, fx_rates)
            if skipped:
                log.warning(
                    "Omitted %d ticker(s): %s",
                    len(skipped), ", ".join(s["ticker"] for s in skipped),
                )
            if unknown_fx:
                log.warning(
                    "Unknown FX (USD fallback) for: %s",
                    ", ".join(f"{s['ticker']} ({s['currency']})" for s in unknown_fx),
                )
            try:
                sp500 = yf.Ticker("^GSPC").history(period="2d")["Close"]
                _hist_cache["^GSPC"] = sp500
            except Exception:
                pass
            record_history(positions, summary, fx_rates)
        except Exception as exc:
            log.error("Data-only run failed: %s", exc)
            sys.exit(1)
        return

    if not args.force and not is_send_saturday():
        msg = (
            "Today is not a fortnight send Saturday. "
            "Use --force to run anyway."
        )
        log.info(msg)
        print(msg)
        return

    log.info("=== Portfolio Agent starting ===")

    try:
        rows = load_portfolio("portfolio.csv")

        log.info("Fetching FX rates…")
        fx_rates = get_fx_rates()

        log.info("Fetching portfolio market data…")
        positions, summary, unknown_fx, skipped = fetch_portfolio_data(rows, fx_rates)
        positions.sort(key=lambda p: p["value_eur"], reverse=True)

        breakdown = value_change_breakdown(summary["total_value_eur"], load_cashflows())
        summary["value_breakdown"] = breakdown
        if breakdown:
            # Supersedes vs-January, which mixes deposits into the change.
            summary["value_vs_jan_pct"] = None
            log.info("Value change since %s: %+.0f EUR (paid in %+.0f, market %+.0f)",
                     breakdown["since"], breakdown["change"],
                     breakdown["invested"], breakdown["market"])
        else:
            vs_jan = get_history_ytd(summary["total_value_eur"])
            summary["value_vs_jan_pct"] = vs_jan
            if vs_jan is not None:
                log.info("Book value vs January snapshot: %+.2f%%", vs_jan)

        if not positions:
            log.error("No position data could be fetched. Aborting.")
            sys.exit(1)

        log.info("Fetching benchmark data…")
        benchmarks = fetch_benchmarks(fx_rates)

        log.info("Fetching portfolio news…")
        port_news = fetch_portfolio_news(positions)
        log.info("Filtering news by relevance…")
        port_news = filter_news_by_relevance(port_news, positions)

        log.info("Fetching general news…")
        gen_news = fetch_general_news()

        log.info("Computing advanced metrics…")
        metrics = compute_advanced_metrics(positions, fx_rates)

        log.info("Computing momentum scores…")
        momentum_data = compute_momentum_scores(positions)

        log.info("Computing look-through exposure…")
        lookthrough_data = compute_lookthrough_exposure(positions)

        log.info("Fetching upcoming events…")
        events = fetch_upcoming_events(positions)

        targets = load_targets()
        allocation = compute_allocation_drift(positions, targets) if targets else []

        if not args.force:
            record_history(positions, summary, fx_rates)
        log.info("Generating charts…")
        charts = generate_charts(positions, summary, benchmarks)

        log.info("Calling Claude for analysis…")
        conviction = load_conviction()
        current_issue = issue_number()
        conviction_prior = {k: v for k, v in conviction.items() if v.get("report", 0) < current_issue}
        ref_prices = recommendation_ref_prices(conviction_prior, positions, fx_rates)
        analysis_raw = call_claude(
            positions, summary, benchmarks, port_news, gen_news, metrics,
            conviction_prior, momentum_data, lookthrough_data, events, allocation, ref_prices,
        )
        _assert_analysis_complete(analysis_raw)
        analysis_body, actionable_raw = split_analysis(analysis_raw)
        actionable_items = parse_actionable(actionable_raw)
        scorecard = conviction_scorecard(conviction_prior, positions, ref_prices)
        if args.preview:
            log.info("Preview — conviction.json left unchanged")
        else:
            save_conviction(conviction, actionable_items, positions, current_issue)

        log.info("Calling Claude for headline…")
        try:
            headline = call_claude_headline(summary, benchmarks, metrics)
        except Exception as exc:
            log.warning("Headline generation failed (%s) — using fallback", exc)
            headline = (
                f"Portfolio moved {summary['fortnight_pct']:+.2f}% this fortnight, "
                f"finishing the year {summary['ytd_pct']:+.2f}% YTD."
            )

        log.info("Building HTML email…")
        chart_cids = {name: f"chart_{name}@portfolio" for name in charts}
        html = build_html_email(
            positions, summary, benchmarks, port_news, gen_news,
            analysis_body, actionable_items, headline,
            chart_cids, metrics, current_issue,
            unknown_fx=unknown_fx,
            skipped=skipped,
            allocation=allocation,
            events=events,
            scorecard=scorecard,
            holdings_as_of=holdings_as_of(),
            unmapped=load_unmapped_isins(),
        )

        if args.preview:
            with open("preview.html", "w", encoding="utf-8") as fh:
                fh.write(html)
            log.info("Preview written to preview.html (%d chart files in %s)",
                     len(charts), tempfile.gettempdir())
            print("Preview written to preview.html.")
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
