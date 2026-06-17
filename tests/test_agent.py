"""Tests for pure functions in agent.py."""
import numpy as np
import pandas as pd
import pytest
from datetime import date as real_date
from unittest.mock import patch

import agent
from agent import (
    issue_number,
    is_second_saturday,
    get_ticker_classification,
    _matches_macro,
    _first_token,
    split_analysis,
    parse_actionable,
    _markdown_to_html,
    _emphasize_numbers,
    _prepare_pie,
    _trailing,
    _trailing_sharpe,
    _trailing_beta,
    _trailing_vol,
    compute_momentum_scores,
    compute_lookthrough_exposure,
)


# ── issue_number ──────────────────────────────────────────────────────────────

def test_issue_number_jan_2025():
    assert issue_number(real_date(2025, 1, 1)) == 1

def test_issue_number_dec_2025():
    assert issue_number(real_date(2025, 12, 1)) == 12

def test_issue_number_jan_2026():
    # Crosses year boundary
    assert issue_number(real_date(2026, 1, 1)) == 13


# ── is_second_saturday ────────────────────────────────────────────────────────
# Jan 2026: Jan 1 = Thursday → first Sat = Jan 3, second Sat = Jan 10

def test_is_second_saturday_true():
    with patch("agent.date") as mock_date:
        mock_date.today.return_value = real_date(2026, 1, 10)
        assert is_second_saturday()

def test_is_second_saturday_false_not_saturday():
    with patch("agent.date") as mock_date:
        mock_date.today.return_value = real_date(2026, 1, 11)  # Sunday
        assert not is_second_saturday()

def test_is_second_saturday_false_first_saturday():
    with patch("agent.date") as mock_date:
        mock_date.today.return_value = real_date(2026, 1, 3)
        assert not is_second_saturday()

def test_is_second_saturday_different_month_start():
    # Feb 2026: Feb 1 = Sunday → second Sat = Feb 14
    with patch("agent.date") as mock_date:
        mock_date.today.return_value = real_date(2026, 2, 14)
        assert is_second_saturday()


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
    monkeypatch.setattr(agent, "_hist_cache", {"AAA": pd.Series(vals)})

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
    monkeypatch.setattr(agent, "_hist_cache", {})
    result = compute_momentum_scores([{"ticker": "ZZZ", "name": "Zeta"}])
    assert result == []

def test_compute_momentum_scores_skips_short_history(monkeypatch):
    short_series = pd.Series(np.linspace(100, 110, 50))  # < 63 days
    monkeypatch.setattr(agent, "_hist_cache", {"AAA": short_series})
    result = compute_momentum_scores([{"ticker": "AAA", "name": "Alpha"}])
    assert result == []

def test_compute_momentum_scores_r6m_falls_back_to_r3m_when_under_126_days(monkeypatch):
    vals = np.linspace(100, 150, 90)  # >= 63, < 126
    monkeypatch.setattr(agent, "_hist_cache", {"AAA": pd.Series(vals)})
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
    monkeypatch.setattr(agent, "_hist_cache", cache)

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
