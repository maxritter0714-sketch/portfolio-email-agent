"""Tests for import_cashflows.py and the value-change breakdown."""
from datetime import date

import pytest

import market
from import_cashflows import daily_cashflows

ACCOUNT = "Trade Republic Verrechnungskonto"


def _row(day, amount, category, account=ACCOUNT, ref=""):
    return {"Buchungsdatum": day, "Betrag": str(amount), "Kategorie": category, "Konto": account,
            "Referenz": ref}


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
        {"date": "2026-10-01", "invested_eur": 150.0, "income_eur": 0.0, "orders": 0},
        {"date": "2026-10-02", "invested_eur": -40.0, "income_eur": 3.5, "orders": 0},
        {"date": "2026-10-03", "invested_eur": 0.0, "income_eur": 0.0, "orders": 0},
    ]


def test_daily_cashflows_counts_only_one_off_orders():
    rows = [
        _row("01.10.2026", -76.0, "Wertpapieranlage", ref="Buy Order: Fluence Energy"),
        _row("01.10.2026", 76.0, "Wertpapieranlage", ref="Sell Order: Grab"),  # nets to 0, still 2 orders
        _row("01.10.2026", -10.0, "Wertpapieranlage", ref="Saving executed: Microsoft"),
        _row("01.10.2026", -10.0, "Wertpapieranlage", ref="Saving Executed: Nvidia"),
        _row("01.10.2026", -0.5, "Wertpapieranlage", ref="Round up: Core MSCI Europe EUR (Acc)"),
        _row("01.10.2026", -1.7, "Wertpapieranlage", ref="Saveback: Core MSCI Europe EUR (Acc)"),
        _row("02.10.2026", 50.0, "Wertpapieranlage", ref="stop sell: Sea (ADR)"),
        _row("02.10.2026", 5.0, "Wertpapieranlage", ref="Warrant exercise: XYZ"),
        _row("02.10.2026", -2.0, "Wertpapieranlage", ref="Preliminary Lump Sum: ETF"),
    ]
    assert [(f["date"], f["orders"]) for f in daily_cashflows(rows, ACCOUNT)] == [
        ("2026-10-01", 2), ("2026-10-02", 1)]


def test_load_cashflows_old_format_has_no_order_count(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("date,invested_eur,income_eur\n2026-10-01,100.0,0.0\n", encoding="utf-8")
    new = tmp_path / "new.csv"
    new.write_text("date,invested_eur,income_eur,orders\n2026-10-01,100.0,0.0,3\n", encoding="utf-8")
    assert market.load_cashflows(old)[0]["orders"] is None
    assert market.load_cashflows(new)[0]["orders"] == 3


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
