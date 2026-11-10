"""Capture the README screenshots from the demo email preview and dashboard.

Render both pages from the synthetic portfolio first, never from real data:

    python tools/make_demo_data.py
    export PORTFOLIO_DATA_DIR=docs/demo PORTFOLIO_OUTPUT_DIR=output/demo
    python agent.py --force --preview
    python dashboard.py
    python tools/readme_screenshots.py

Needs Playwright (`pip install playwright && playwright install chromium`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"

# (file name, text the shot starts at, text it stops at; None = page edge)
EMAIL_SHOTS = [
    ("email-overview.png", None, "II."),
    ("email-history.png", "II.", "Fig. 2"),
    ("email-analysis.png", "III.", "(3) News"),
    ("email-actionable.png", "VI.", "VIII."),
]
DASHBOARD_SHOTS = [
    ("dashboard-record.png", "Your actual record", "Today's portfolio"),
    ("dashboard-backtest.png", "Today's portfolio", "Returns by period"),
    ("dashboard-structure.png", "Structure", "Correlation"),
    ("dashboard-montecarlo.png", "Monte Carlo", "Recommendation track record"),
]


def _text_top(page: Page, prefix: str | None, below: float = 0.0) -> float:
    """Page y of the first visible text starting with `prefix` below `below` (0 for None)."""
    if prefix is None:
        return 0.0
    top = page.evaluate(
        """([p, below]) => {
             const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
             for (let n = w.nextNode(); n; n = w.nextNode()) {
               const el = n.parentElement;
               if (['SCRIPT', 'STYLE'].includes(el.tagName)) continue;
               if (!n.textContent.trim().startsWith(p)) continue;
               const y = el.getBoundingClientRect().top + window.scrollY;
               if (y > below && el.offsetParent !== null) return y;
             }
             return null; }""",
        [prefix, below],
    )
    if top is None:
        raise SystemExit(f"No text starting with {prefix!r} found")
    return float(top)


def _shoot(page: Page, shots: list[tuple[str, str | None, str | None]], pad: float = 12) -> None:
    height = page.evaluate("document.documentElement.scrollHeight")
    width = page.viewport_size["width"]
    for name, start, end in shots:
        top = _text_top(page, start)
        y0 = max(0.0, top - (pad * 2 if start else 0))
        y1 = _text_top(page, end, below=top) - pad if end else height
        page.screenshot(path=IMAGES / name, full_page=True,
                        clip=dict(x=0, y=y0, width=width, height=y1 - y0))
        print(f"  {name}: {y1 - y0:.0f}px tall")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", default=str(ROOT / "output" / "demo" / "preview.html"))
    parser.add_argument("--dashboard", default=str(ROOT / "output" / "demo" / "dashboard.html"))
    args = parser.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        # Email clients render at about 700px; the dashboard is a desktop page.
        for path, shots, width in ((args.email, EMAIL_SHOTS, 720),
                                   (args.dashboard, DASHBOARD_SHOTS, 1280)):
            page = browser.new_page(viewport=dict(width=width, height=900), device_scale_factor=2,
                                    color_scheme="light")
            page.goto(Path(path).resolve().as_uri(), wait_until="networkidle")
            page.wait_for_timeout(1500)  # let Plotly finish drawing
            print(Path(path).name)
            _shoot(page, shots)
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
