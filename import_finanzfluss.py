#!/usr/bin/env python3
"""
import_finanzfluss.py

Converts a Finanzfluss portfolio export CSV into the portfolio.csv format
expected by agent.py.

The core of this script is the ISIN_TO_TICKER map — a hardcoded lookup table
that resolves ISIN codes to Yahoo Finance ticker symbols. It covers a broad
range of US equities, European stocks, Asian ADRs, and UCITS ETFs. For any
ISIN not in the map, the script falls back to a live yfinance search.

This script is designed specifically for Finanzfluss exports and expects the
following column names (as exported by Finanzfluss):

    Name        → portfolio.csv: name
    ISIN        → used to resolve the Yahoo Finance ticker
    Anzahl      → portfolio.csv: shares
    Kaufpreis   → portfolio.csv: avg_buy_price  (EUR, German locale format)
    Typ         → filter only — positions of type "Zertifikate/OS" are skipped

If you use a different broker, adapt the column name constants at the top of
this file (COLUMNS_READ) to match your export format.

Columns intentionally ignored (never written to portfolio.csv):
    Aktueller Kurs   – live price from Finanzfluss (stale; yfinance is authoritative)
    Aktueller Wert   – live value (derived; recalculated at runtime)
    WKN, Währung, Wechselkurs, Region, Sektor – not used

Output: portfolio.csv with exactly: ticker, name, shares, avg_buy_price
All live price data is fetched from yfinance at runtime by agent.py.

Usage:
    python import_finanzfluss.py                        # default paths
    python import_finanzfluss.py --input my_export.csv --output portfolio.csv
    python import_finanzfluss.py --quiet                # suppress per-row output
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# ── Columns extracted from the Finanzfluss CSV (read these, nothing else) ─────
COLUMNS_READ = ("Name", "ISIN", "Anzahl", "Kaufpreis")

# ── Output columns written to portfolio.csv (exactly these, in this order) ────
OUTPUT_COLUMNS = ("ticker", "name", "shares", "avg_buy_price")

# ── ISIN → Yahoo Finance ticker ───────────────────────────────────────────────
# Covers the full universe of common instruments. For anything not here the
# script falls back to a live yfinance search, then warns if that also fails.
ISIN_TO_TICKER: dict[str, str] = {
    # ── US Equities ──────────────────────────────────────────────────────────
    "US0378331005": "AAPL",   # Apple
    "US5949181045": "MSFT",   # Microsoft
    "US67066G1040": "NVDA",   # NVIDIA
    "US88160R1014": "TSLA",   # Tesla
    "US92826C8394": "V",      # Visa
    "US22788C1053": "CRWD",   # Crowdstrike
    "US0231351067": "AMZN",   # Amazon
    "US02079K3059": "GOOGL",  # Alphabet (A)
    "US30303M1027": "META",   # Meta Platforms (A)
    "US1491231015": "CAT",    # Caterpillar
    "US18915M1071": "NET",    # Cloudflare
    "US6974351057": "PANW",   # Palo Alto Networks
    "US0567521085": "BIDU",   # Baidu (ADR)
    "US01609W1027": "BABA",   # Alibaba (ADR)
    "US09857L1089": "BKNG",   # Booking Holdings
    "US69608A1088": "PLTR",   # Palantir Technologies
    "US83406F1021": "SOFI",   # SoFi Technologies
    "US8887871080": "TOST",   # Toast
    "US8740391003": "TSM",    # TSMC (ADR)
    "US00724F1012": "ADBE",   # Adobe
    "US0079031078": "AMD",    # AMD
    "US7731211089": "RKLB",   # Rocket Lab
    "US21873S1087": "CRWV",   # CoreWeave
    "US81141R1005": "SE",     # Sea Ltd (ADR)
    "US21037T1097": "CEG",    # Constellation Energy
    "US7811541090": "RBRK",   # Rubrik
    "US5324571083": "LLY",    # Eli Lilly
    "US36828A1016": "GEV",    # GE Vernova
    "US81762P1021": "NOW",    # ServiceNow
    "US6098391054": "MPWR",   # Monolithic Power Systems
    "US03831W1080": "APP",    # AppLovin
    "US0320951017": "APH",    # Amphenol
    "US98956A1051": "ZETA",   # Zeta Global Holdings
    "US92537N1081": "VRT",    # Vertiv
    "US0404132054": "ANET",   # Arista Networks
    "US05605H1005": "BWXT",   # BWX Technologies
    "US58733R1023": "MELI",   # MercadoLibre
    "US90353T1007": "UBER",   # Uber
    "US98422D1054": "XPEV",   # Xpeng (ADR)
    "US47215P1066": "JD",     # JD.com (ADR)
    "US70450Y1038": "PYPL",   # PayPal
    "US58933Y1055": "MRK",    # Merck & Co.
    "US78409V1044": "SPGI",   # S&P Global
    "US55354G1004": "MSCI",   # MSCI Inc.
    "US91324P1021": "UNH",    # UnitedHealth
    "US0258161092": "AXP",    # American Express
    "US46625H1005": "JPM",    # JPMorgan Chase
    "US5024311095": "LHX",    # L3Harris Technologies
    "US9256521090": "VICI",   # Vici Properties
    "US4612021034": "INTU",   # Intuit
    "US9224751084": "VEEV",   # Veeva Systems
    "US8308791024": "SKYW",   # SkyWest
    "US0494681010": "TEAM",   # Atlassian (A)
    "US4435731009": "HUBS",   # HubSpot
    "US8334451098": "SNOW",   # Snowflake
    "US19260Q1076": "COIN",   # Coinbase
    "US4330001060": "HIMS",   # Hims & Hers Health
    "US86800U3023": "SMCI",   # Super Micro Computer
    "US0846707026": "BRK-B",  # Berkshire Hathaway (B)
    "US76954A1034": "RIVN",   # Rivian Automotive
    "US81730H1095": "S",      # SentinelOne
    "US5949724083": "MSTR",   # Strategy (MicroStrategy)
    "US8725401090": "TJX",    # TJX Companies
    "US87151X1019": "SYM",    # Symbotic
    "US12468P1049": "AI",     # C3.ai
    "US36118L1061": "FUTU",   # Futu Holdings (ADR)
    "US7223041028": "PDD",    # PDD Holdings / Temu (ADR)
    "US1380357048": "CGC",    # Canopy Growth
    "US8308791024": "SKYW",   # SkyWest
    "US1491231015": "CAT",    # Caterpillar (duplicate guard)
    "US4000027842": "MU",     # Micron Technology
    "US5801351017": "MCD",    # McDonald's
    "US0231351067": "AMZN",   # Amazon (duplicate guard)
    "US88160R1014": "TSLA",   # Tesla (duplicate guard)

    # ── Canadian Equities (US-listed) ────────────────────────────────────────
    "CA13321L1085": "CCJ",    # Cameco
    "CA65340P1062": "NXE",    # NexGen Energy
    "CA1380357048": "CGC",    # Canopy Growth

    # ── Australian Equities ──────────────────────────────────────────────────
    "AU0000185993": "IREN",   # IREN (Iris Energy, NASDAQ)

    # ── Cayman / Caribbean (US-listed) ───────────────────────────────────────
    "KYG6683N1034": "NU",     # Nu Holdings (NYSE)
    "KYG4124C1096": "GRAB",   # Grab Holdings (NASDAQ)
    "KYG5479M1050": "LI",     # Li Auto (ADR)
    "KYG9830T1067": "1810.HK",  # Xiaomi (Hong Kong)

    # ── Netherlands / Nordic (US-listed or local) ─────────────────────────────
    "NL0009805522": "NBIS",   # Nebius Group (NASDAQ)
    "NL0013056914": "ESTC",   # Elastic NV (NYSE)
    "NL0011585146": "RACE",   # Ferrari (NYSE)
    "DK0062498333": "NVO",    # Novo-Nordisk (ADR via NYSE)
    "DK0060094928": "DNNGY",  # Orsted (ADR) — local: ORSTED.CO
    "NO0003067902": "HEX.OL", # Hexagon Composites (Oslo)

    # ── Swiss Equities ───────────────────────────────────────────────────────
    "CH0334081137": "CRSP",   # CRISPR Therapeutics (NASDAQ)
    "CH1335392721": "GALD.SW", # Galderma (SIX)

    # ── German Equities ──────────────────────────────────────────────────────
    "DE0005677108": "ELG.DE",  # Elmos Semiconductor
    "DE000A1K0235": "SMHN.DE", # Suess MicroTec
    "DE0007276503": "YSN.DE",  # secunet Security Networks
    "DE0007030009": "RHM.DE",  # Rheinmetall
    "DE0007664039": "VOW3.DE", # Volkswagen (Vz.)
    "DE000A0TGJ55": "VAR1.DE", # VARTA
    "DE0005190003": "BMW.DE",  # BMW
    "DE000A3E5A59": "SYB.DE",  # Synbiotic
    "DE0007568578": "F3C.DE",  # SFC Energy
    "DE0008404005": "ALV.DE",  # Allianz

    # ── French Equities ──────────────────────────────────────────────────────
    "FR0000121014": "MC.PA",   # LVMH

    # ── UK Equities ──────────────────────────────────────────────────────────
    "GB00BP6S8Z30": "ONT.L",   # Oxford Nanopore Technologies

    # ── Asian Equities ───────────────────────────────────────────────────────
    "CNE100006WS8": "3750.HK",    # CATL (Hong Kong listing)
    "CNE100000296": "BYDDY",      # BYD (US ADR)
    "HK0285041858": "285.HK",     # BYD Electronic (Hong Kong)
    "US98422E1038": "XERS",       # Xeris Biopharma (check; may not be this ISIN)

    # ── South American Equities ──────────────────────────────────────────────
    "US58733R1023": "MELI",   # MercadoLibre (duplicate guard)

    # ── Clean Energy ─────────────────────────────────────────────────────────
    "US34379V1035": "FLNC",   # Fluence Energy

    # ── ETFs — iShares UCITS (XETRA .DE / London .L) ─────────────────────────
    "IE00B5BMR087": "SXR8.DE",   # iShares Core S&P 500 UCITS ETF
    "IE00B4K48X80": "IMAE.AS",   # iShares Core MSCI Europe UCITS ETF (Euronext Amsterdam)
    "IE00B3WJKG14": "IUIT.L",    # iShares S&P 500 Info Tech UCITS ETF
    "IE00BM67HK77": "WHCS.L",    # iShares MSCI World Health Care UCITS ETF
    "IE00B3CNHG25": "IAUG.L",    # iShares Gold Producers UCITS ETF
    "IE00B4ND3602": "IGLN.L",    # iShares Physical Gold ETC
    "IE00B4NCWG09": "ISLN.L",    # iShares Physical Silver ETC
    "IE000KCS7J59": "HEMA.L",    # HSBC MSCI Emerging Markets UCITS ETF (USD, LSE)
    "IE00BQT3WG13": "CNYA.L",    # iShares MSCI China A UCITS ETF
    "IE00BKWQ0J47": "STQ.PA",    # SPDR MSCI Europe Industrials UCITS ETF (Euronext Paris)
    "IE00BGV5VN51": "WTAI.L",    # WisdomTree AI & Big Data UCITS ETF
    "IE000YYE6WK5": "DFND.L",    # WisdomTree Defense UCITS ETF
    "IE000OJ5TQP4": "ADEF.L",    # WisdomTree Future of Defence ETF
    "IE00BYZK4552": "RBOT.L",    # iShares Automation & Robotics UCITS ETF
    "IE00B1XNHC34": "INRG.L",    # iShares Global Clean Energy UCITS ETF
    "IE00BKPSFC54": "WQDI.L",    # iShares MSCI World Quality Dividend UCITS ETF
    "IE000L2SA8K5": "QNEW.L",    # Invesco NASDAQ-100 Equal Weight UCITS ETF
    "IE00B53SZB19": "CNDX.L",    # iShares NASDAQ-100 UCITS ETF (large cap)
    "IE000BI8OT95": "XDWD.DE",   # iShares Core MSCI World UCITS ETF
    "IE000O5M6XO1": "ARKG",      # ARK Genomic Revolution ETF (check ISIN)
    "IE000YU9K6K2": "JEDI.L",    # VanEck Space Innovators UCITS ETF (rebranded/relisted from YODA)
    "IE00BMC38736": "SEMI.L",     # VanEck Semiconductor UCITS ETF

    # ── ETFs — Lyxor / Amundi UCITS ──────────────────────────────────────────
    "LU1781541179": "LCWD.L",    # Lyxor Core MSCI World (DR) UCITS ETF
    "LU1781541252": "LCJP.L",    # Lyxor Core MSCI Japan (DR) UCITS ETF
    "LU1681041973": "CD9.PA",    # Amundi MSCI Europe High Dividend Factor UCITS ETF (ex-Lyxor LYEH)
    "LU1681044720": "540J.DE",   # Amundi MSCI Switzerland UCITS ETF (ex-Lyxor, XETRA)
    "LU0392494562": "CEMR.DE",   # Lyxor MSCI Emerging Markets UCITS ETF

    # ── ETFs — Invesco / Other ────────────────────────────────────────────────
    "IE00BMFKG444": "XNAS.L",    # Xtrackers NASDAQ-100 UCITS ETF 1C (USD, LSE)
    "IE00BZ0G8C04": "IJPH.L",    # iShares MSCI Japan EUR Hedged UCITS ETF
    "IE00B53SZB19": "CNDX.L",    # iShares NASDAQ-100 (acc) — dup guard
}

# Types to exclude (warrants, options, certificates)
EXCLUDED_TYPES = {"Zertifikate/OS"}


def parse_german_float(raw: str) -> float:
    """
    Convert a German-locale number string to a Python float.
    '1.234,56' → 1234.56 | '3,41' → 3.41 | '0' → 0.0
    """
    s = raw.strip().strip('"').strip()
    if not s or s == "N/A":
        return 0.0
    # Remove thousands separator (period), swap decimal comma → period
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def try_yfinance_search(isin: str, name: str) -> str | None:
    """
    Attempt to resolve an unmapped ISIN via yfinance's Search API.
    Returns a ticker symbol or None if nothing was found.
    Requires yfinance >= 0.2.37; older versions will silently skip.
    """
    try:
        import yfinance as yf

        # 1. Try the ISIN directly
        try:
            s = yf.Search(isin, max_results=1)
            quotes = getattr(s, "quotes", [])
            if quotes:
                sym = quotes[0].get("symbol", "")
                if sym:
                    return sym
        except Exception:
            pass

        time.sleep(0.3)  # be polite to Yahoo's API

        # 2. Try the human-readable name (strip trailing parenthetical junk)
        short_name = name.split("(")[0].strip()
        try:
            s = yf.Search(short_name, max_results=1)
            quotes = getattr(s, "quotes", [])
            if quotes:
                sym = quotes[0].get("symbol", "")
                if sym:
                    return sym
        except Exception:
            pass

    except ImportError:
        pass

    return None


def resolve_ticker(isin: str, name: str, verbose: bool = True) -> str | None:
    """
    Return a Yahoo Finance ticker for the given ISIN, or None if unresolvable.
    Priority: hardcoded map → yfinance search.
    """
    # 1. Hardcoded map (fast, reliable)
    if isin in ISIN_TO_TICKER:
        return ISIN_TO_TICKER[isin]

    # 2. yfinance Search fallback
    if verbose:
        print(f"  [search] ISIN {isin} not in hardcoded map — trying yfinance search for '{name}'…")
    ticker = try_yfinance_search(isin, name)
    if ticker:
        if verbose:
            print(f"  [search] Found: {ticker}")
        return ticker

    return None


def read_finanzfluss_csv(filepath: str) -> list[dict]:
    """Parse the Finanzfluss export and return cleaned rows."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: '{filepath}' not found.")
        sys.exit(1)

    with open(filepath, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    return rows


def convert(
    input_path: str = "investments.csv",
    output_path: str = "portfolio.csv",
    verbose: bool = True,
) -> tuple[int, list[str]]:
    """
    Convert a Finanzfluss export to portfolio.csv.

    Reads exactly the columns in COLUMNS_READ (Name, ISIN, Anzahl, Kaufpreis).
    Writes exactly the columns in OUTPUT_COLUMNS (ticker, name, shares, avg_buy_price).
    No Finanzfluss price or valuation data ever reaches the output file.

    Returns (number_of_holdings_written, list_of_warning_strings).
    """
    rows = read_finanzfluss_csv(input_path)

    if verbose:
        print(f"Read {len(rows)} rows from '{input_path}'.")

    holdings: list[dict] = []
    warnings: list[str] = []

    for row in rows:
        # Read only the four columns declared in COLUMNS_READ.
        # All other Finanzfluss columns (Aktueller Kurs, Aktueller Wert, WKN,
        # Währung, Wechselkurs, Region, Sektor, …) are never accessed here.
        name = row.get("Name", "").strip()
        isin = row.get("ISIN", "").strip()
        anzahl_raw = row.get("Anzahl", "0")
        kaufpreis_raw = row.get("Kaufpreis", "0")

        # Typ is read solely for filtering (Zertifikate/OS exclusion) and is
        # never forwarded to the output — not part of COLUMNS_READ intentionally.
        typ = row.get("Typ", "").strip()

        # ── Filter ────────────────────────────────────────────────────────────
        if typ in EXCLUDED_TYPES:
            continue  # skip warrants/certificates

        anzahl = parse_german_float(anzahl_raw)
        if anzahl <= 0:
            continue  # skip positions with zero holdings

        kaufpreis = parse_german_float(kaufpreis_raw)

        # ── ISIN → Yahoo ticker ───────────────────────────────────────────────
        ticker = resolve_ticker(isin, name, verbose=verbose)

        if ticker is None:
            msg = (
                f"WARNING: Could not map ISIN '{isin}' ({name}) to a Yahoo Finance ticker. "
                f"Add it manually to portfolio.csv or to ISIN_TO_TICKER in import_finanzfluss.py."
            )
            warnings.append(msg)
            if verbose:
                print(f"  {msg}")
            continue

        holdings.append(
            {
                "ticker": ticker,       # from ISIN map — the only price-data source is yfinance
                "name": name,           # from Finanzfluss "Name"
                "shares": f"{anzahl:.6g}",       # from Finanzfluss "Anzahl"
                "avg_buy_price": f"{kaufpreis:.4f}",  # from Finanzfluss "Kaufpreis" (EUR)
                # NOTE: Aktueller Kurs and Aktueller Wert are not included here.
                #       Live prices are fetched exclusively from yfinance at runtime.
            }
        )

    # ── Write output — strictly OUTPUT_COLUMNS, nothing else ─────────────────
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OUTPUT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(holdings)

    if verbose:
        print(f"\nWrote {len(holdings)} active holdings to '{output_path}'.")
        if warnings:
            print(f"\n{len(warnings)} ISIN(s) could not be mapped — add them manually:")
            for w in warnings:
                print(f"  • {w}")

    return len(holdings), warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Finanzfluss export CSV to portfolio.csv for agent.py"
    )
    parser.add_argument(
        "--input",
        default="investments.csv",
        help="Path to the Finanzfluss export CSV (default: investments.csv)",
    )
    parser.add_argument(
        "--output",
        default="portfolio.csv",
        help="Path for the output portfolio.csv (default: portfolio.csv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-row output; only print summary and warnings",
    )
    args = parser.parse_args()

    count, warnings = convert(args.input, args.output, verbose=not args.quiet)

    if warnings:
        sys.exit(1)  # non-zero exit so callers know there were unmapped ISINs


if __name__ == "__main__":
    main()


