"""Tests for import_cashflows.py and the value-change breakdown."""
from datetime import date

import pytest

import market
from import_cashflows import daily_cashflows

ACCOUNT = "Trade Republic Verrechnungskonto"


def _row(day, amount, category, account=ACCOUNT):
    return {"Buchungsdatum": day, "Betrag": str(amount), "Kategorie": category, "Konto": account}


def test_daily_cashflows_nets_buys_sells_and_income():
    rows = [
        _row("01.10.2026", -100.0, "Wertpapieranlage"),
        _row("01.10.2026", -50.0, "Wertpapieranlage"),
        _row("02.10.2026", 40.0, "Wertpapieranlage"),      # sale
        _row("02.10.2026", 3.5, "Kapitaleinkommen"),
        _row("03.10.2026", -12.0, "Lebensmittel & Getränke"),  # private spending
        _row("01.10.2026", -999.0, "Wertpapieranlage", account="Krypto"),
    ]
    assert daily_cashflows(rows, ACCOUNT) == [
        {"date": "2026-10-01", "invested_eur": 150.0, "income_eur": 0.0},
        {"date": "2026-10-02", "invested_eur": -40.0, "income_eur": 3.5},
        {"date": "2026-10-03", "invested_eur": 0.0, "income_eur": 0.0},
    ]


def test_breakdown_splits_change_into_paid_in_and_market():
    history = [
        {"date": "2026-06-14", "total_value_eur": "30000"},
        {"date": "2026-07-01", "total_value_eur": "31000"},
    ]
    flows = [
        {"date": "2026-06-14", "invested_eur": 500.0, "income_eur": 0.0},  # on baseline day: excluded
        {"date": "2026-08-01", "invested_eur": 1000.0, "income_eur": 20.0},
        {"date": "2026-10-23", "invested_eur": 0.0, "income_eur": 0.0},
    ]
    b = market.value_change_breakdown(32500.0, flows, history, today=date(2026, 10, 28))
    assert b["since"] == date(2026, 6, 14)
    assert b["change"] == pytest.approx(2500.0)
    assert b["invested"] == pytest.approx(1000.0)
    assert b["market"] == pytest.approx(1500.0)
    assert b["income"] == pytest.approx(20.0)
    assert b["flows_as_of"] == date(2026, 10, 23)


def test_breakdown_uses_first_row_of_current_year():
    history = [
        {"date": "2026-12-26", "total_value_eur": "30000"},
        {"date": "2027-01-02", "total_value_eur": "31000"},
    ]
    flows = [{"date": "2027-01-05", "invested_eur": 100.0, "income_eur": 0.0}]
    b = market.value_change_breakdown(31500.0, flows, history, today=date(2027, 1, 9))
    assert b["since"] == date(2027, 1, 2)
    assert b["market"] == pytest.approx(400.0)


def test_breakdown_none_without_cashflows():
    assert market.value_change_breakdown(1.0, [], [{"date": "2026-01-01", "total_value_eur": "1"}]) is None
