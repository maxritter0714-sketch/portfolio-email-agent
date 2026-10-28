"""Tests for pure functions in import_portfolio.py."""
import pytest

from import_portfolio import parse_german_float, _pick_ticker_from_figi


# ── parse_german_float ────────────────────────────────────────────────────────

def test_parse_german_float_standard():
    assert parse_german_float("1.234,56") == pytest.approx(1234.56)

def test_parse_german_float_zero():
    assert parse_german_float("0") == 0.0

def test_parse_german_float_empty():
    assert parse_german_float("") == 0.0

def test_parse_german_float_na():
    assert parse_german_float("N/A") == 0.0

def test_parse_german_float_quoted():
    assert parse_german_float('"1.234,56"') == pytest.approx(1234.56)


# ── _pick_ticker_from_figi ────────────────────────────────────────────────────

def test_pick_ticker_us_no_suffix():
    assert _pick_ticker_from_figi("US0378331005", [{"exchCode": "US", "ticker": "AAPL"}]) == "AAPL"

def test_pick_ticker_de_xetra_suffix():
    assert _pick_ticker_from_figi("DE0007164600", [{"exchCode": "GY", "ticker": "SAP"}]) == "SAP.DE"

def test_pick_ticker_ie_prefers_xetra_over_london():
    # IE ISIN (UCITS ETF) — priority is GY before LN
    candidates = [{"exchCode": "LN", "ticker": "VUSA"}, {"exchCode": "GY", "ticker": "VUSA"}]
    assert _pick_ticker_from_figi("IE00B3XXRP09", candidates) == "VUSA.DE"

def test_pick_ticker_fallback_to_known_exchange():
    # Unknown ISIN prefix but known exchange in _EXCH_TO_SUFFIX
    assert _pick_ticker_from_figi("XX1234567890", [{"exchCode": "GY", "ticker": "XYZ"}]) == "XYZ.DE"

def test_pick_ticker_no_candidates():
    assert _pick_ticker_from_figi("US0378331005", []) is None

def test_pick_ticker_unknown_exchange():
    assert _pick_ticker_from_figi("US0378331005", [{"exchCode": "ZZ", "ticker": "FOO"}]) is None


# ── convert: unmapped ISINs ───────────────────────────────────────────────────

def test_convert_writes_unmapped_isins(tmp_path, monkeypatch):
    import json
    import import_portfolio

    export = tmp_path / "export.csv"
    export.write_text(
        "Name,ISIN,Anzahl,Kaufpreis,Typ\n"
        "Mapped Co,US0000000001,\"1,5\",\"10,00\",Aktie\n"
        "Unknown AG,DE000TEST001,2,\"5,00\",Aktie\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(import_portfolio, "ISIN_TO_TICKER", {"US0000000001": "MAP"})
    monkeypatch.setattr(import_portfolio, "bulk_openfigi_lookup", lambda isins, verbose=True: {})
    monkeypatch.setattr(import_portfolio, "try_yfinance_search", lambda isin, name: None)

    out = tmp_path / "portfolio.csv"
    count, warnings = import_portfolio.convert(str(export), str(out), verbose=False)
    assert count == 1 and len(warnings) == 1
    unmapped = json.loads((tmp_path / "unmapped_isins.json").read_text(encoding="utf-8"))
    assert unmapped == [{"isin": "DE000TEST001", "name": "Unknown AG"}]
