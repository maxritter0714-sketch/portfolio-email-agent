"""Tests for the dashboard analytics and payload helpers (pure functions only)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

import market
from analytics import (
    buy_hold_returns,
    capture_ratios,
    concentration,
    contributions,
    cumulative_flows,
    drawdown_stats,
    index_from_returns,
    monthly_returns,
    period_return,
    period_start,
    portfolio_xirr,
    risk_contributions,
    risk_stats,
    to_weekly,
    xirr,
)
from dashboard import _json_ready, render_html
from market import region_exposure, sector_exposure


def _days(n, start="2024-01-01"):
    return pd.bdate_range(start, periods=n)


# ── buy_hold_returns ─────────────────────────────────────────────────────────

def test_buy_hold_returns_weights_by_value():
    idx = _days(2)
    prices = pd.DataFrame({"A": [10.0, 11.0], "B": [10.0, 10.0]}, index=idx)
    ret, _ = buy_hold_returns(prices, pd.Series({"A": 1.0, "B": 3.0}))
    # Value 40 → 41: +2.5 %
    assert ret.iloc[0] == pytest.approx(0.025)


def test_buy_hold_returns_late_listing_does_not_jump():
    idx = _days(3)
    prices = pd.DataFrame({"A": [10.0, 10.0, 10.0], "B": [np.nan, 50.0, 50.0]}, index=idx)
    ret, coverage = buy_hold_returns(prices, pd.Series({"A": 1.0, "B": 1.0}))
    assert ret.tolist() == [0.0, 0.0]
    # B is 50 of today's 60
    assert coverage.iloc[0] == pytest.approx(10 / 60)
    assert coverage.iloc[-1] == pytest.approx(1.0)


def test_buy_hold_returns_carries_prices_over_holidays():
    idx = _days(3)
    prices = pd.DataFrame({"A": [10.0, np.nan, 11.0], "B": [10.0, 10.0, 10.0]}, index=idx)
    ret, _ = buy_hold_returns(prices, pd.Series({"A": 1.0, "B": 1.0}))
    assert ret.iloc[0] == pytest.approx(0.0)
    assert ret.iloc[1] == pytest.approx(0.05)


# ── Drawdown and periods ─────────────────────────────────────────────────────

def test_drawdown_stats_max_current_and_longest():
    idx = pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-21", "2024-01-31", "2024-02-10"])
    index = pd.Series([100.0, 80.0, 90.0, 100.0, 95.0], index=idx)
    dd = drawdown_stats(index)
    assert dd["max_dd"] == pytest.approx(-0.2)
    assert dd["max_dd_date"] == "2024-01-11"
    assert dd["current_dd"] == pytest.approx(-0.05)
    assert dd["longest_days"] == 20


def test_period_start_ytd_is_last_day_of_prior_year():
    assert period_start(pd.Timestamp("2026-10-29"), "YTD") == pd.Timestamp("2025-12-31")
    assert period_start(pd.Timestamp("2026-10-29"), "3M") == pd.Timestamp("2026-07-29")


def test_period_return_none_when_history_too_short():
    idx = pd.to_datetime(["2024-06-01", "2024-07-01"])
    index = pd.Series([100.0, 110.0], index=idx)
    assert period_return(index, pd.Timestamp("2024-06-15")) == pytest.approx(0.10)
    assert period_return(index, pd.Timestamp("2024-01-01")) is None


def test_monthly_returns_compounds_and_totals():
    idx = pd.to_datetime(["2024-01-15", "2024-01-16", "2024-02-15"])
    ret = pd.Series([0.10, 0.10, -0.50], index=idx)
    m = monthly_returns(ret)
    assert m["years"] == [2024]
    assert m["grid"][0][0] == pytest.approx(0.21)
    assert m["grid"][0][1] == pytest.approx(-0.50)
    assert m["grid"][0][2] is None
    assert m["totals"][0] == pytest.approx(1.21 * 0.5 - 1)


# ── Risk ─────────────────────────────────────────────────────────────────────

def test_to_weekly_compounds_within_week_and_skips_gaps():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-15"])
    weekly = to_weekly(pd.Series([0.10, 0.10, 0.05], index=idx))
    assert weekly.iloc[0] == pytest.approx(0.21)
    assert np.isnan(weekly.iloc[1])
    assert weekly.iloc[2] == pytest.approx(0.05)


def test_capture_ratios_for_double_leverage():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"])
    bench = pd.Series([0.02, -0.01, 0.03, -0.02], index=idx)
    up, down = capture_ratios(bench * 2, bench)
    assert up == pytest.approx(2.0, rel=0.05)
    assert down == pytest.approx(2.0, rel=0.05)


def test_risk_stats_portfolio_equal_to_benchmark():
    rng = np.random.default_rng(1)
    idx = _days(300)
    r = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    stats = risk_stats(r, r, r, risk_free=0.0)
    assert stats["beta"] == pytest.approx(1.0)
    assert stats["correlation"] == pytest.approx(1.0)
    assert stats["tracking_error"] == pytest.approx(0.0)
    assert stats["information_ratio"] is None
    assert stats["alpha"] == pytest.approx(0.0, abs=1e-12)


def test_risk_stats_needs_sixty_days():
    r = pd.Series([0.01] * 30, index=_days(30))
    assert risk_stats(r, r, r, 0.0) == {}


# ── Per position and structure ──────────────────────────────────────────────

def test_contributions_sum_to_portfolio_return():
    idx = _days(3)
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [20.0, 19.0, 18.0]}, index=idx)
    shares = pd.Series({"A": 2.0, "B": 1.0})
    c = contributions(prices, shares, idx[0])
    assert c.loc["A", "ret"] == pytest.approx(0.2)
    assert c["contrib"].sum() == pytest.approx((2 * 12 + 18) / (2 * 10 + 20) - 1)


def test_contributions_before_history_is_empty():
    prices = pd.DataFrame({"A": [10.0]}, index=_days(1, "2024-06-03"))
    assert contributions(prices, pd.Series({"A": 1.0}), pd.Timestamp("2020-01-01")).empty


def test_risk_contributions_sum_to_one_and_ratio_one_for_identical_names():
    rng = np.random.default_rng(2)
    idx = _days(260)
    r = rng.normal(0, 0.01, len(idx))
    returns = pd.DataFrame({"A": r, "B": r}, index=idx)
    share, vol, ratio = risk_contributions(returns, pd.Series({"A": 0.25, "B": 0.75}))
    assert share.sum() == pytest.approx(1.0)
    assert share["B"] == pytest.approx(0.75)
    assert vol > 0
    assert ratio == pytest.approx(1.0)


def test_concentration_effective_n():
    c = concentration(pd.Series({"A": 0.5, "B": 0.25, "C": 0.25}))
    assert c["hhi"] == pytest.approx(0.375)
    assert c["effective_n"] == pytest.approx(1 / 0.375)
    assert c["count"] == 3


def test_exposures_spread_etf_sectors():
    positions = [
        dict(value_eur=100.0, sector="ETF", region="Global", sector_weights={"Technology": 0.6, "Energy": 0.4}),
        dict(value_eur=50.0, sector="Technology", region="North America", sector_weights={}),
    ]
    assert sector_exposure(positions) == pytest.approx({"Technology": 110.0, "Energy": 40.0})
    assert region_exposure(positions) == {"Global": 100.0, "North America": 50.0}


# ── Real track record ───────────────────────────────────────────────────────

def test_xirr_one_year_ten_percent():
    assert xirr([(date(2024, 1, 1), -100.0), (date(2025, 1, 1), 110.0)]) == pytest.approx(0.0998, abs=1e-3)


def test_xirr_none_without_sign_change():
    assert xirr([(date(2024, 1, 1), 100.0), (date(2025, 1, 1), 110.0)]) is None


def test_portfolio_xirr_counts_income_and_sales():
    flows = [
        dict(date="2024-01-01", invested_eur=100.0, income_eur=0.0),
        dict(date="2024-07-01", invested_eur=-50.0, income_eur=5.0),
    ]
    # Paying in 100 and getting 55 back after half a year plus 60 at the end is clearly positive.
    assert portfolio_xirr(flows, 60.0, date(2025, 1, 1)) > 0.1


def test_cumulative_flows_running_totals():
    flows = [
        dict(date="2024-01-01", invested_eur=100.0, income_eur=0.0),
        dict(date="2024-02-01", invested_eur=-30.0, income_eur=2.0),
    ]
    assert cumulative_flows(flows)[-1] == dict(date="2024-02-01", invested=70.0, income=2.0)


# ── Payload and rendering ───────────────────────────────────────────────────

def test_json_ready_replaces_nan_and_numpy():
    out = _json_ready({"a": float("nan"), "b": np.float64(1.23456789), "c": [np.int64(3), float("inf")]})
    assert out == {"a": None, "b": 1.234568, "c": [3, None]}


def test_render_html_escapes_script_close():
    html = render_html({"name": "</script><b>"}, "<script>const D = /*__DATA__*/null;</script>")
    assert "</script><b>" not in html
    assert '<\\/script>' in html


def test_index_from_returns_compounds():
    idx = _days(2)
    assert index_from_returns(pd.Series([0.1, 0.1], index=idx)).iloc[-1] == pytest.approx(121.0)


# ── market FX period cache ──────────────────────────────────────────────────

def test_to_eur_close_long_period_uses_its_own_cache(monkeypatch):
    idx = _days(2)
    short_fx = pd.Series([0.5, 0.5], index=idx)
    long_fx = pd.Series([0.8, 0.8], index=idx)
    monkeypatch.setattr(market, "_fx_hist_cache", {"USD": short_fx, "USD:7y": long_fx})
    close = pd.Series([10.0, 10.0], index=idx)
    assert market._to_eur_close(close, "USD", {"USD": 0.9}).iloc[0] == pytest.approx(5.0)
    assert market._to_eur_close(close, "USD", {"USD": 0.9}, fx_period="7y").iloc[0] == pytest.approx(8.0)
