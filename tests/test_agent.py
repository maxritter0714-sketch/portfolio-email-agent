"""Tests for pure functions across the agent modules."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from datetime import date as real_date

import market
from claude_report import (
    _assert_analysis_complete,
    _value_vs_jan_line,
    match_action_ticker,
    parse_actionable,
    save_conviction,
    split_analysis,
)
from email_html import _emphasize_numbers, _eur_signed, _markdown_to_html, build_html_email
from market import (
    _FX_FALLBACKS,
    _FX_PAIRS,
    _prepare_pie,
    _to_eur_close,
    _window_pct,
    get_history_ytd,
    get_ticker_classification,
    is_send_saturday,
    issue_number,
    to_eur_rate,
)
from metrics import (
    _trailing,
    _trailing_beta,
    _trailing_sharpe,
    _trailing_vol,
    compute_advanced_metrics,
    compute_lookthrough_exposure,
    compute_momentum_scores,
    risk_free_rate,
)
from news import (
    _extract_text_from_response,
    _first_token,
    _matches_macro,
    _query_name_batches,
    _tag_article_tickers,
)


# ── issue_number ──────────────────────────────────────────────────────────────

def test_issue_number_epoch():
    assert issue_number(real_date(2025, 1, 11)) == 1

def test_issue_number_same_fortnight_off_week():
    assert issue_number(real_date(2025, 1, 18)) == 1

def test_issue_number_second_fortnight():
    assert issue_number(real_date(2025, 1, 25)) == 2

def test_issue_number_before_epoch_is_one():
    assert issue_number(real_date(2025, 1, 1)) == 1


# ── is_send_saturday ──────────────────────────────────────────────────────────

def test_is_send_saturday_epoch():
    assert is_send_saturday(real_date(2025, 1, 11))

def test_is_send_saturday_false_not_saturday():
    assert not is_send_saturday(real_date(2025, 1, 12))  # Sunday

def test_is_send_saturday_false_off_week():
    assert not is_send_saturday(real_date(2025, 1, 18))

def test_is_send_saturday_two_weeks_later():
    assert is_send_saturday(real_date(2025, 1, 25))

def test_is_send_saturday_before_epoch():
    assert not is_send_saturday(real_date(2025, 1, 4))  # Saturday before № 1


# ── get_ticker_classification ─────────────────────────────────────────────────

def test_classification_etf_sector():
    sector, _ = get_ticker_classification("VUSA", {"quoteType": "ETF", "longName": "Vanguard S&P 500 ETF"})
    assert sector == "ETF"

def test_classification_etf_sp500_region():
    _, region = get_ticker_classification("VUSA", {"quoteType": "ETF", "longName": "Vanguard S&P 500 ETF"})
    assert region == "North America"

def test_classification_etf_no_longname_defaults_global():
    _, region = get_ticker_classification("XYZ", {"quoteType": "ETF"})
    assert region == "Global"

def test_classification_stock_sector_and_region():
    sector, region = get_ticker_classification("AAPL", {"quoteType": "EQUITY", "sector": "Technology", "country": "United States"})
    assert sector == "Technology"
    assert region == "North America"

def test_classification_stock_germany_bucketed():
    _, region = get_ticker_classification("SAP", {"quoteType": "EQUITY", "sector": "Technology", "country": "Germany"})
    assert region == "Europe"

def test_classification_stock_missing_fields():
    sector, region = get_ticker_classification("XYZ", {})
    assert sector == "Other"
    assert region == "Other"


# ── _matches_macro ────────────────────────────────────────────────────────────

def test_matches_macro_economy_keyword():
    assert _matches_macro("Fed signals rate cut amid slowing GDP growth")

def test_matches_macro_tech_keyword():
    assert _matches_macro("New AI breakthrough could reshape semiconductor industry")

def test_matches_macro_excluded_sport_keyword():
    # "sport" in title triggers exclusion even if macro keyword present
    assert not _matches_macro("Rate your favourite sports team in our new poll")

def test_matches_macro_no_keywords():
    assert not _matches_macro("Local bakery wins award for best croissant")


# ── _first_token ──────────────────────────────────────────────────────────────

def test_first_token_simple():
    assert _first_token("Apple Inc.") == "apple"

def test_first_token_skips_stop_words():
    assert _first_token("The Boeing Company") == "boeing"

def test_first_token_empty_string():
    assert _first_token("") == ""


def test_query_name_batches_splits_when_over_budget():
    names = [f"Company Number {i} Holdings PLC" for i in range(20)]
    batches = _query_name_batches(names, max_name_chars=90)
    assert len(batches) > 1
    for clause in batches:
        assert len(clause) <= 90
    joined = " ".join(batches)
    assert "Company Number 0 Holdings PLC" in joined
    assert "Company Number 19 Holdings PLC" in joined


def test_query_name_batches_skips_name_longer_than_budget():
    huge = "X" * 80
    batches = _query_name_batches([huge, "Nvidia"], max_name_chars=20)
    assert batches == ['"Nvidia"']


def test_tag_skips_generic_first_token_apple():
    article = {"headline": "Apple pie brand hits the stock market", "description": ""}
    positions = [{"ticker": "AAPL", "name": "Apple Inc."}]
    assert _tag_article_tickers(article, positions) == []


def test_tag_skips_generic_first_token_meta():
    article = {"headline": "A meta analysis of inflation and the stock market", "description": ""}
    positions = [{"ticker": "META", "name": "Meta Platforms Inc."}]
    assert _tag_article_tickers(article, positions) == []


def test_tag_meta_when_company_name_present():
    article = {"headline": "Meta Platforms reports quarterly earnings", "description": ""}
    positions = [{"ticker": "META", "name": "Meta Platforms Inc."}]
    assert _tag_article_tickers(article, positions) == ["META"]


def test_tag_nvidia_by_distinctive_first_token():
    article = {"headline": "Nvidia raises full-year guidance", "description": ""}
    positions = [{"ticker": "NVDA", "name": "NVIDIA Corporation"}]
    assert _tag_article_tickers(article, positions) == ["NVDA"]


def test_tag_by_ticker_symbol():
    article = {"headline": "NVDA hits a record close", "description": ""}
    positions = [{"ticker": "NVDA", "name": "NVIDIA Corporation"}]
    assert _tag_article_tickers(article, positions) == ["NVDA"]


def test_tag_by_ticker_root_when_suffix_present():
    article = {"headline": "ASML ships more EUV tools", "description": ""}
    positions = [{"ticker": "ASML.AS", "name": "ASML Holding NV"}]
    assert _tag_article_tickers(article, positions) == ["ASML.AS"]


# ── split_analysis ────────────────────────────────────────────────────────────

SAMPLE_ANALYSIS = """\
## (1) Portfolio Performance Summary
Portfolio gained 3.2% this fortnight.

## (2) Benchmark Comparison
Outperformed S&P 500 by 1.5%.

## (3) News & Market Context
Fed pause boosted growth stocks.

## (4) Actionable Suggestions
[TRIM] Reduce NVDA — Concentration risk elevated."""

def test_split_analysis_body_has_sections_1_to_3():
    body, _ = split_analysis(SAMPLE_ANALYSIS)
    assert "(1) Portfolio Performance Summary" in body
    assert "(2) Benchmark Comparison" in body
    assert "(3) News & Market Context" in body

def test_split_analysis_actionable_not_in_body():
    body, _ = split_analysis(SAMPLE_ANALYSIS)
    assert "Actionable" not in body

def test_split_analysis_extracts_actionable_text():
    _, actionable = split_analysis(SAMPLE_ANALYSIS)
    assert "[TRIM]" in actionable


# ── parse_actionable ──────────────────────────────────────────────────────────

def test_parse_actionable_parses_title_and_detail():
    items = parse_actionable("[TRIM] Reduce NVDA — Concentration too high.")
    assert len(items) == 1
    assert items[0]["action"] == "TRIM"
    assert items[0]["title"] == "Reduce NVDA"
    assert "Concentration" in items[0]["detail"]

def test_parse_actionable_multiple_actions():
    text = "[TRIM] Trim A — Detail A.\n[EXIT] Exit B — Detail B.\n[RESEARCH] Check C — Detail C."
    items = parse_actionable(text)
    assert [i["action"] for i in items] == ["TRIM", "EXIT", "RESEARCH"]

def test_parse_actionable_caps_at_five():
    lines = "\n".join(f"[TRIM] Item {i} — Detail {i}." for i in range(8))
    assert len(parse_actionable(lines)) == 5

def test_parse_actionable_empty():
    assert parse_actionable("") == []

def test_parse_actionable_case_insensitive():
    items = parse_actionable("[trim] Lower weight — Too concentrated.")
    assert items[0]["action"] == "TRIM"


_POS_NVDA = {"ticker": "NVDA", "name": "NVIDIA Corporation", "price_eur": 120.0}
_POS_AMD = {"ticker": "AMD", "name": "Advanced Micro Devices", "price_eur": 90.0}


def test_match_action_ticker_by_symbol():
    item = {"title": "Reduce NVDA", "detail": "Too large."}
    assert match_action_ticker(item, [_POS_NVDA, _POS_AMD]) == "NVDA"


def test_match_action_ticker_by_company_name():
    item = {"title": "Reduce Nvidia", "detail": "Concentration too high."}
    assert match_action_ticker(item, [_POS_NVDA, _POS_AMD]) == "NVDA"


def test_match_action_ticker_name_in_detail_only():
    item = {"title": "Trim the winner", "detail": "Nvidia still 22% of the book."}
    assert match_action_ticker(item, [_POS_NVDA, _POS_AMD]) == "NVDA"


def test_match_action_ticker_ambiguous_names_returns_none():
    item = {"title": "Reduce Nvidia and AMD", "detail": ""}
    assert match_action_ticker(item, [_POS_NVDA, _POS_AMD]) is None


def test_match_action_ticker_ignores_short_name_token():
    sap = {"ticker": "SAP.DE", "name": "SAP SE", "price_eur": 180.0}
    item = {"title": "Review the software names", "detail": ""}
    assert match_action_ticker(item, [sap]) is None


def test_save_conviction_writes_name_match(tmp_path, monkeypatch):
    path = tmp_path / "conviction.json"
    monkeypatch.setattr("claude_report._CONVICTION_PATH", path)
    conviction: dict = {}
    save_conviction(
        conviction,
        [{"action": "TRIM", "title": "Reduce Nvidia", "detail": "Too big."}],
        [_POS_NVDA],
        issue_num=12,
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["NVDA"]["action"] == "TRIM"
    assert saved["NVDA"]["report"] == 12


# ── _markdown_to_html ─────────────────────────────────────────────────────────

def test_markdown_h2_produces_h3():
    result = _markdown_to_html("## Section Heading")
    assert "<h3" in result and "Section Heading" in result

def test_markdown_bullet_produces_li():
    result = _markdown_to_html("- First item")
    assert "<li" in result and "First item" in result

def test_markdown_bold():
    assert "<strong>bold</strong>" in _markdown_to_html("This is **bold** text.")

def test_markdown_link():
    result = _markdown_to_html("[Click here](https://example.com)")
    assert 'href="https://example.com"' in result and "Click here" in result

def test_markdown_escapes_ampersand():
    result = _markdown_to_html("AT&T reports earnings.")
    assert "&amp;" in result and "&T" not in result

def test_markdown_escapes_angle_brackets():
    result = _markdown_to_html("Price > 100 and < 200.")
    assert "&lt;" in result and "&gt;" in result


# ── _emphasize_numbers ────────────────────────────────────────────────────────

def test_emphasize_numbers_wraps_first_only():
    result = _emphasize_numbers("Up 5% then down 3%")
    assert result.count("<em") == 1

def test_emphasize_numbers_no_numbers():
    assert "<em" not in _emphasize_numbers("No numbers here")


# ── _prepare_pie ──────────────────────────────────────────────────────────────

def test_prepare_pie_max_slices_creates_others():
    items = [("A", 40.0), ("B", 30.0), ("C", 20.0), ("D", 10.0)]
    labels, sizes = _prepare_pie(items, max_slices=2)
    assert labels[:2] == ["A", "B"]
    assert "Others" in labels
    assert sizes[-1] == pytest.approx(30.0)

def test_prepare_pie_no_others_when_all_fit():
    labels, _ = _prepare_pie([("A", 50.0), ("B", 50.0)], max_slices=5)
    assert "Others" not in labels

def test_prepare_pie_min_pct_aggregates_small():
    items = [("Big", 80.0), ("Small", 5.0), ("Tiny", 2.0)]
    labels, _ = _prepare_pie(items, min_pct=10.0)
    assert "Big" in labels
    assert "Others" in labels
    assert "Small" not in labels


# ── _trailing / rolling metrics ─────────────────────────────────────────────

def test_trailing_slices_last_n():
    arr = np.arange(10, dtype=float)
    result = _trailing(arr, 3)
    assert list(result) == [7.0, 8.0, 9.0]

def test_trailing_falls_back_to_full_array_when_shorter():
    arr = np.arange(5, dtype=float)
    result = _trailing(arr, 10)
    assert list(result) == list(arr)

def test_trailing_sharpe_positive_returns_gives_positive_sharpe():
    pr = np.full(60, 0.01)
    pr = pr + np.array([0.001 if i % 2 == 0 else -0.001 for i in range(60)])
    sharpe = _trailing_sharpe(pr, 30, risk_free=0.0, tdays=252)
    assert sharpe > 0

def test_trailing_sharpe_window_larger_than_array_uses_full_array():
    pr = np.full(20, 0.01) + np.array([0.001 if i % 2 == 0 else -0.001 for i in range(20)])
    full = _trailing_sharpe(pr, 90, risk_free=0.0, tdays=252)
    direct = _trailing_sharpe(pr, len(pr), risk_free=0.0, tdays=252)
    assert full == pytest.approx(direct)

def test_trailing_beta_equal_series_is_one():
    pr = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.012])
    beta = _trailing_beta(pr, pr, window=5)
    assert beta == pytest.approx(1.0)

def test_trailing_beta_window_larger_than_array_uses_full_array():
    pr = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    sr = np.array([0.008, -0.018, 0.012, -0.004, 0.018])
    full = _trailing_beta(pr, sr, window=90)
    direct = _trailing_beta(pr, sr, window=len(pr))
    assert full == pytest.approx(direct)

def test_trailing_beta_zero_variance_benchmark_returns_none():
    pr = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    sr = np.full(5, 0.0)
    assert _trailing_beta(pr, sr, window=5) is None

def test_trailing_vol_matches_manual_calculation():
    pr = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    expected = np.std(pr, ddof=1) * np.sqrt(252)
    assert _trailing_vol(pr, window=5, tdays=252) == pytest.approx(expected)

def test_trailing_vol_window_larger_than_array_uses_full_array():
    pr = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    full = _trailing_vol(pr, window=90, tdays=252)
    direct = _trailing_vol(pr, window=len(pr), tdays=252)
    assert full == pytest.approx(direct)


# ── compute_momentum_scores ─────────────────────────────────────────────────

def test_compute_momentum_scores_happy_path(monkeypatch):
    length = 130
    vals = np.linspace(50, 300, length)
    vals[0] = 100.0
    vals[length - 126] = 105.0
    vals[length - 63] = 150.0
    vals[length - 21] = 180.0
    vals[-1] = 200.0
    monkeypatch.setattr(market, "_hist_cache", {"AAA": pd.Series(vals)})

    result = compute_momentum_scores([{"ticker": "AAA", "name": "Alpha Co"}])

    assert len(result) == 1
    d = result[0]
    assert d["r1m"] == pytest.approx(200 / 180 - 1)
    assert d["r3m"] == pytest.approx(200 / 150 - 1)
    assert d["r6m"] == pytest.approx(200 / 105 - 1)
    assert d["r12m"] == pytest.approx(1.0)
    expected_score = 0.35 * d["r3m"] + 0.35 * d["r6m"] + 0.30 * d["r12m"]
    assert d["score"] == pytest.approx(expected_score)

def test_compute_momentum_scores_skips_missing_from_cache(monkeypatch):
    monkeypatch.setattr(market, "_hist_cache", {})
    result = compute_momentum_scores([{"ticker": "ZZZ", "name": "Zeta"}])
    assert result == []

def test_compute_momentum_scores_skips_short_history(monkeypatch):
    short_series = pd.Series(np.linspace(100, 110, 50))  # < 63 days
    monkeypatch.setattr(market, "_hist_cache", {"AAA": short_series})
    result = compute_momentum_scores([{"ticker": "AAA", "name": "Alpha"}])
    assert result == []

def test_compute_momentum_scores_r6m_falls_back_to_r3m_when_under_126_days(monkeypatch):
    vals = np.linspace(100, 150, 90)  # >= 63, < 126
    monkeypatch.setattr(market, "_hist_cache", {"AAA": pd.Series(vals)})
    result = compute_momentum_scores([{"ticker": "AAA", "name": "Alpha"}])
    d = result[0]
    assert d["r6m"] == pytest.approx(d["r3m"])

def test_compute_momentum_scores_quartiles_span_1_to_4_and_sort_descending(monkeypatch):
    def make_series(growth: float, length: int = 130) -> pd.Series:
        start, end = 100.0, 100.0 * (1 + growth)
        vals = np.full(length, start)
        vals[0] = start
        vals[length - 126] = end
        vals[length - 63] = end
        vals[length - 21] = end
        vals[-1] = end
        return pd.Series(vals)

    growths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    cache = {f"T{i}": make_series(g) for i, g in enumerate(growths)}
    positions = [{"ticker": f"T{i}", "name": f"Ticker {i}"} for i in range(8)]
    monkeypatch.setattr(market, "_hist_cache", cache)

    result = compute_momentum_scores(positions)

    assert [d["ticker"] for d in result] == [f"T{i}" for i in reversed(range(8))]
    expected_quartiles = {f"T{i}": i // 2 + 1 for i in range(8)}
    for d in result:
        assert d["quartile"] == expected_quartiles[d["ticker"]]


# ── compute_lookthrough_exposure ────────────────────────────────────────────

def test_lookthrough_amplified_when_etfs_add_20pct_or_more_to_direct_position():
    positions = [
        {"ticker": "ETF", "name": "Some ETF", "value_eur": 1000.0,
         "holdings_weights": {"NVDA": 0.20}},
        {"ticker": "NVDA", "name": "Nvidia", "value_eur": 100.0,
         "holdings_weights": {}},
    ]
    result = compute_lookthrough_exposure(positions)

    nvda = next(d for d in result if d["ticker"] == "NVDA")
    assert nvda["kind"] == "amplified"
    assert nvda["direct_eur"] == pytest.approx(100.0)
    assert nvda["effective_eur"] == pytest.approx(300.0)
    assert nvda["hidden_eur"] == pytest.approx(200.0)
    # The ETF itself has zero hidden exposure (no look-through into itself)
    assert all(d["ticker"] != "ETF" for d in result)

def test_lookthrough_excludes_directly_held_position_amplified_under_20pct():
    positions = [
        {"ticker": "ETF", "name": "Some ETF", "value_eur": 1000.0,
         "holdings_weights": {"NVDA": 0.05}},  # adds 50 on top of 1000 direct = 5%
        {"ticker": "NVDA", "name": "Nvidia", "value_eur": 1000.0,
         "holdings_weights": {}},
    ]
    result = compute_lookthrough_exposure(positions)
    assert all(d["ticker"] != "NVDA" for d in result)

def test_lookthrough_hidden_kind_for_ticker_not_held_directly():
    positions = [
        {"ticker": "ETF", "name": "Some ETF", "value_eur": 1000.0,
         "holdings_weights": {"MSFT": 0.05}},
    ]
    result = compute_lookthrough_exposure(positions)

    msft = next(d for d in result if d["ticker"] == "MSFT")
    assert msft["kind"] == "hidden"
    assert msft["direct_eur"] == pytest.approx(0.0)
    assert msft["effective_eur"] == pytest.approx(50.0)
    assert msft["hidden_eur"] == pytest.approx(50.0)

def test_lookthrough_excludes_hidden_exposure_below_threshold():
    positions = [
        {"ticker": "ETF", "name": "Some ETF", "value_eur": 10_000.0,
         "holdings_weights": {"TINY": 0.002}},  # 0.2% of total, under the 0.3% floor
    ]
    result = compute_lookthrough_exposure(positions)
    assert all(d["ticker"] != "TINY" for d in result)

def test_lookthrough_empty_for_zero_total_value():
    positions = [{"ticker": "AAA", "name": "Alpha", "value_eur": 0.0, "holdings_weights": {}}]
    assert compute_lookthrough_exposure(positions) == []

def test_lookthrough_sorted_descending_by_hidden_eur():
    positions = [
        {"ticker": "ETF1", "name": "ETF One", "value_eur": 1000.0,
         "holdings_weights": {"BIG": 0.10, "SMALL": 0.02}},
    ]
    result = compute_lookthrough_exposure(positions)
    assert [d["ticker"] for d in result] == ["BIG", "SMALL"]


# ── to_eur_rate / FX coverage ────────────────────────────────────────────────

def test_fx_pairs_include_nordic_and_cad():
    for code in ("DKK", "SEK", "NOK", "CAD", "AUD"):
        assert code in _FX_PAIRS
        assert code in _FX_FALLBACKS

def test_to_eur_rate_known_currency_no_fallback():
    rates = {"EUR": 1.0, "USD": 0.9, "DKK": 0.13}
    rate, used_fallback = to_eur_rate("DKK", rates)
    assert rate == pytest.approx(0.13)
    assert used_fallback is False

def test_to_eur_rate_unknown_currency_uses_usd_fallback():
    rates = {"EUR": 1.0, "USD": 0.9}
    rate, used_fallback = to_eur_rate("XYZ", rates)
    assert rate == pytest.approx(0.9)
    assert used_fallback is True

def test_to_eur_rate_gbx_alias():
    rates = {"EUR": 1.0, "USD": 0.9, "GBp": 0.0117, "GBX": 0.0117}
    rate, used_fallback = to_eur_rate("GBX", rates)
    assert rate == pytest.approx(0.0117)
    assert used_fallback is False


def _minimal_email(**kwargs) -> str:
    defaults = dict(
        positions=[],
        summary=dict(
            total_value_eur=10_000.0,
            fortnight_pct=1.0,
            monthly_pct=2.0,
            weekly_pct=0.5,
            ytd_pct=8.0,
            portfolio_yield_pct=1.2,
        ),
        benchmarks=[],
        port_news=[],
        gen_news=[],
        analysis_body="## (1) Summary\nSteady.",
        actionable_items=[],
        headline="Up 1% this fortnight",
        chart_cids={},
        metrics={},
        issue_num=1,
    )
    defaults.update(kwargs)
    return build_html_email(**defaults)

def test_unknown_fx_appears_in_footer_notes():
    html_out = _minimal_email(
        unknown_fx=[{"ticker": "NOVOB.CO", "name": "Novo", "currency": "DKK"}],
    )
    assert "Unknown FX converted via USD fallback: NOVOB.CO (DKK)." in html_out

def test_skipped_tickers_appear_in_footer_notes():
    html_out = _minimal_email(
        skipped=[{"ticker": "NOPE", "name": "Nope Inc", "reason": "404"}],
    )
    assert "Omitted (no price data): NOPE." in html_out

def test_actionable_title_and_detail_are_html_escaped():
    html_out = _minimal_email(
        actionable_items=[{
            "action": "TRIM",
            "title": "<script>alert(1)</script>",
            "detail": "Foo & Bar",
        }],
    )
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "Foo &amp; Bar" in html_out


def test_masthead_keeps_price_ytd_and_omits_vs_january_when_missing():
    html_out = _minimal_email()
    assert ">YTD<" in html_out or ">YTD</div>" in html_out
    assert "+8.00%" in html_out
    assert "Vs January" not in html_out
    assert "Includes cash in/out" not in html_out


def test_masthead_shows_vs_january_beside_price_ytd():
    html_out = _minimal_email(
        summary=dict(
            total_value_eur=10_000.0,
            fortnight_pct=1.0,
            monthly_pct=2.0,
            weekly_pct=0.5,
            ytd_pct=8.0,
            value_vs_jan_pct=20.0,
            portfolio_yield_pct=1.2,
        ),
    )
    assert "+8.00%" in html_out
    assert "+20.00%" in html_out
    assert "Vs January" in html_out
    assert "Includes cash in/out" in html_out


def test_value_vs_jan_line_empty_when_missing():
    assert _value_vs_jan_line({"ytd_pct": 8.0}) == ""


def test_value_vs_jan_line_warns_not_twr():
    line = _value_vs_jan_line({"value_vs_jan_pct": 20.0})
    assert "+20.00%" in line
    assert "not a time-weighted return" in line


def test_get_history_ytd_uses_first_january_row(tmp_path, monkeypatch):
    class FakeDate(real_date):
        @classmethod
        def today(cls):
            return cls(2026, 8, 27)

    hist = tmp_path / "history.csv"
    hist.write_text(
        "date,issue,total_value_eur,sp500_close\n"
        "2026-01-04,1,100000,5000\n"
        "2026-01-11,1,101000,5010\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(market, "_HISTORY_PATH", hist)
    monkeypatch.setattr(market, "date", FakeDate)
    assert get_history_ytd(120_000.0) == pytest.approx(20.0)


def test_get_history_ytd_none_without_january(tmp_path, monkeypatch):
    class FakeDate(real_date):
        @classmethod
        def today(cls):
            return cls(2026, 8, 27)

    hist = tmp_path / "history.csv"
    hist.write_text(
        "date,issue,total_value_eur,sp500_close\n"
        "2026-02-07,3,110000,5100\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(market, "_HISTORY_PATH", hist)
    monkeypatch.setattr(market, "date", FakeDate)
    assert get_history_ytd(120_000.0) is None


# ── EUR conversion for beta / alpha ───────────────────────────────────────────

def test_to_eur_close_leaves_eur_unchanged():
    idx = pd.bdate_range("2025-01-02", periods=3)
    native = pd.Series([10.0, 11.0, 12.0], index=idx)
    out = _to_eur_close(native, "EUR", {"EUR": 1.0})
    assert list(out) == [10.0, 11.0, 12.0]


def test_to_eur_close_usd_uses_daily_fx(monkeypatch):
    idx = pd.bdate_range("2025-01-02", periods=3)
    close = pd.Series([100.0, 100.0, 100.0], index=idx)
    fx = pd.Series([0.8, 0.9, 1.0], index=idx)
    monkeypatch.setattr(market, "_fx_hist_cache", {"USD": fx})
    out = _to_eur_close(close, "USD", {"USD": 0.85})
    assert list(out) == pytest.approx([80.0, 90.0, 100.0])


def test_to_eur_close_gbx_converts_pence_then_gbp_fx(monkeypatch):
    idx = pd.bdate_range("2025-01-02", periods=2)
    close = pd.Series([1000.0, 1100.0], index=idx)
    fx = pd.Series([1.2, 1.2], index=idx)
    monkeypatch.setattr(market, "_fx_hist_cache", {"GBP": fx})
    out = _to_eur_close(close, "GBX", {"GBp": 0.012})
    assert list(out) == pytest.approx([12.0, 13.2])


def test_beta_near_zero_when_usd_holding_is_flat_in_eur(monkeypatch):
    n = 80
    idx = pd.bdate_range("2025-01-02", periods=n)
    fx = pd.Series(np.linspace(0.8, 1.2, n), index=idx)
    usd_stock = 100.0 / fx
    spx = pd.Series(100.0, index=idx)
    monkeypatch.setattr(market, "_fx_hist_cache", {"USD": fx})
    monkeypatch.setattr(market, "_hist_cache", {"AAA": usd_stock, "^GSPC": spx})
    positions = [{
        "ticker": "AAA", "name": "Alpha", "value_eur": 1000.0,
        "pl_pct": 1.0, "currency": "USD",
    }]
    result = compute_advanced_metrics(positions, {"USD": 1.0, "EUR": 1.0})
    assert result["beta"] == pytest.approx(0.0, abs=0.05)


def test_metrics_eur_benchmark_note_in_footer():
    html_out = _minimal_email(metrics={"sharpe": 1.0, "beta": 1.0, "alpha": 0.0})
    assert "S&P 500 in euro terms" in html_out
    assert "euro returns" in html_out
    assert "Above cost" in html_out


def test_window_pct_follows_daily_fx_when_usd_price_is_flat(monkeypatch):
    idx = pd.bdate_range("2025-01-02", periods=8)
    usd = pd.Series(100.0, index=idx)
    fx = pd.Series([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0], index=idx)
    monkeypatch.setattr(market, "_fx_hist_cache", {"USD": fx})
    eur = _to_eur_close(usd, "USD", {"USD": 0.9})
    weekly = _window_pct(eur, 6)
    assert weekly == pytest.approx(25.0)


def test_risk_free_rate_default(monkeypatch):
    monkeypatch.delenv("RISK_FREE_RATE", raising=False)
    assert risk_free_rate() == pytest.approx(0.035)


def test_risk_free_rate_from_env(monkeypatch):
    monkeypatch.setenv("RISK_FREE_RATE", "0.04")
    assert risk_free_rate() == pytest.approx(0.04)


def test_risk_free_rate_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("RISK_FREE_RATE", "nope")
    assert risk_free_rate() == pytest.approx(0.035)


# ── _extract_text_from_response ───────────────────────────────────────────────

def test_extract_text_from_response_uses_text_attribute():
    response = SimpleNamespace(content=[SimpleNamespace(text="hello world")])
    assert _extract_text_from_response(response) == "hello world"


def test_extract_text_from_response_empty_content_returns_empty_string():
    response = SimpleNamespace(content=[])
    assert _extract_text_from_response(response) == ""


# ── _eur_signed ────────────────────────────────────────────────────────────────

def test_eur_signed_negative_sign_before_symbol():
    assert _eur_signed(-254.0) == "-€254"


def test_eur_signed_positive_sign_before_symbol():
    assert _eur_signed(254.0) == "+€254"


def test_eur_signed_zero_is_positive():
    assert _eur_signed(0.0) == "+€0"


# ── _assert_analysis_complete ───────────────────────────────────────────────────

def test_assert_analysis_complete_accepts_sentence_ending_in_period():
    _assert_analysis_complete("This is a complete sentence.")


def test_assert_analysis_complete_accepts_action_prefixed_last_line():
    _assert_analysis_complete("Some prose.\n[ACTION] Reduce NVDA — Too concentrated.")


def test_assert_analysis_complete_raises_on_truncated_last_line():
    with pytest.raises(RuntimeError, match="truncated"):
        _assert_analysis_complete("This sentence just cuts off mid")


def test_assert_analysis_complete_raises_on_empty_text():
    with pytest.raises(RuntimeError, match="empty"):
        _assert_analysis_complete("   ")
