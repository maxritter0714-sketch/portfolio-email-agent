#!/usr/bin/env python3
"""
Portfolio Email Agent
Fetches market data, news, and Claude AI analysis, then sends an editorial-style
HTML portfolio report via Gmail. Collects data every Saturday; sends the email
every other Saturday (every 14 days from 11 Jan 2025). Bypass with --force.
"""

import argparse
import base64
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

from dotenv import load_dotenv

from claude_report import (
    _assert_analysis_complete,
    append_recommendations,
    assert_report_sections,
    call_claude,
    call_claude_headline,
    parse_actionable,
    split_analysis,
    track_record,
)
from email_html import build_html_email
from paths import OUTPUT_DIR
from market import (
    append_history,
    last_trading_day,
    compute_allocation_drift,
    fetch_benchmarks,
    fetch_portfolio_data,
    fetch_upcoming_events,
    generate_charts,
    get_fx_rates,
    get_history_ytd,
    history_blockers,
    holdings_as_of,
    is_send_saturday,
    issue_number,
    lifetime_return,
    load_portfolio,
    load_cashflows,
    load_targets,
    load_unmapped_isins,
    next_portfolio_index,
    recommendation_prices,
    sp500_index_close,
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
OUTPUT_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "agent.log"),
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


def inline_chart_images(html: str, chart_files: dict[str, str]) -> str:
    """Swap cid: chart references for data URIs so a browser can show them."""
    for name, filepath in chart_files.items():
        try:
            with open(filepath, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
        except OSError as exc:
            log.warning("Could not inline chart %s: %s", name, exc)
            continue
        html = html.replace(f"cid:chart_{name}@portfolio", f"data:image/png;base64,{data}")
    return html


# ── History ───────────────────────────────────────────────────────────────────
def pricing_day() -> date:
    """The trading day a run's prices belong to: the S&P 500's last session.

    Holdings are valued at their close on or before this day, so the result
    does not depend on when the run starts or on which markets have reopened.
    """
    return last_trading_day(sp500_index_close())


def record_history(
    positions: list[dict],
    summary: dict,
    fx_rates: dict[str, float],
    skipped: list[dict],
    unknown_fx: list[dict],
) -> list[str]:
    """Append the latest trading day's data point, including the deposit-free portfolio index.

    Writes nothing when the run's data is incomplete and returns the reasons
    (empty when the data was fit to record).
    """
    source = "scheduled" if os.environ.get("GITHUB_ACTIONS") == "true" else "manual"
    sp500 = sp500_index_close()
    blockers = history_blockers(positions, skipped, unknown_fx, sp500)
    if blockers:
        log.error("History row not written, data incomplete: %s", "; ".join(blockers))
        return blockers
    day = pricing_day()
    port_index = next_portfolio_index(positions, fx_rates, today=day)
    sp500_eur = None
    if sp500 is not None and not sp500.empty:
        sp500_eur = float(sp500.iloc[-1]) * fx_rates.get("USD", 0.0) or None
    if append_history(summary, issue_number(day), source=source,
                      port_index=port_index, sp500_eur=sp500_eur, day=day):
        log.info("History updated for %s — total value EUR %.0f, index %.2f (source=%s)",
                 day, summary["total_value_eur"], port_index, source)
    return []


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
            rows = load_portfolio()
            fx_rates = get_fx_rates()
            positions, summary, unknown_fx, skipped = fetch_portfolio_data(
                rows, fx_rates, as_of=pricing_day(),
            )
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
            blockers = record_history(positions, summary, fx_rates, skipped, unknown_fx)
        except Exception as exc:
            log.error("Data-only run failed: %s", exc)
            sys.exit(1)
        if blockers:
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
        rows = load_portfolio()

        log.info("Fetching FX rates…")
        fx_rates = get_fx_rates()

        log.info("Fetching portfolio market data…")
        positions, summary, unknown_fx, skipped = fetch_portfolio_data(
            rows, fx_rates, as_of=pricing_day(),
        )
        positions.sort(key=lambda p: p["value_eur"], reverse=True)

        cashflows = load_cashflows()
        breakdown = value_change_breakdown(summary["total_value_eur"], cashflows)
        summary["value_breakdown"] = breakdown
        lifetime = lifetime_return(summary["total_value_eur"], cashflows)
        summary["lifetime"] = lifetime
        if lifetime:
            log.info("Gain since %s: %+.0f EUR (%+.2f%%) on %.0f EUR paid in",
                     lifetime["since"], lifetime["gain"], lifetime["pct"], lifetime["invested"])
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
        momentum_data = compute_momentum_scores(positions, fx_rates)

        log.info("Computing look-through exposure…")
        lookthrough_data = compute_lookthrough_exposure(positions)

        log.info("Fetching upcoming events…")
        events = fetch_upcoming_events(positions)

        targets = load_targets()
        allocation = compute_allocation_drift(positions, targets) if targets else []

        if not args.force:
            record_history(positions, summary, fx_rates, skipped, unknown_fx)
        log.info("Generating charts…")
        charts = generate_charts(positions, summary, benchmarks)

        log.info("Calling Claude for analysis…")
        current_issue = issue_number()
        rec_prices, world_close = recommendation_prices(positions, fx_rates)
        scorecard = track_record(positions, rec_prices, world_close, before_issue=current_issue)
        analysis_raw = call_claude(
            positions, summary, benchmarks, port_news, gen_news, metrics,
            scorecard, momentum_data, lookthrough_data, events, allocation,
        )
        _assert_analysis_complete(analysis_raw)
        analysis_body, actionable_raw = split_analysis(analysis_raw)
        actionable_items = parse_actionable(actionable_raw)
        assert_report_sections(analysis_body, actionable_items)
        if args.preview:
            log.info("Preview — recommendations.csv left unchanged")
        else:
            append_recommendations(actionable_items, positions, current_issue)

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
            with open(OUTPUT_DIR / "preview.html", "w", encoding="utf-8") as fh:
                fh.write(inline_chart_images(html, charts))
            log.info("Preview written to %s (%d chart files in %s)", OUTPUT_DIR / "preview.html",
                     len(charts), tempfile.gettempdir())
            print(f"Preview written to {OUTPUT_DIR / 'preview.html'}.")
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
