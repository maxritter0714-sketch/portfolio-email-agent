#!/usr/bin/env python3
"""
import_cashflows.py

Reduces a Finanzfluss account-transactions export to daily portfolio cash
flows in cashflows.csv. Only two categories are kept, summed per day:

  invested_eur -- money moved into securities (buys minus sells)
  income_eur   -- dividends and interest paid to the cash account

Everything else in the export (private spending, transfers, counterparties)
is dropped, so cashflows.csv is safe to upload to the private data repo
while the raw export stays on this machine.

Usage:
    python import_cashflows.py --input finanzfluss_transactions.csv
    python import_cashflows.py --input finanzfluss_transactions.csv --account "Trade Republic Verrechnungskonto"
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_ACCOUNT = "Trade Republic Verrechnungskonto"
INVESTED_CATEGORY = "Wertpapieranlage"
INCOME_CATEGORY = "Kapitaleinkommen"
OUTPUT_COLUMNS = ("date", "invested_eur", "income_eur")


def daily_cashflows(rows: list[dict], account: str = DEFAULT_ACCOUNT) -> list[dict]:
    """Per-day invested and income totals for one account, oldest first.

    Finanzfluss books a buy as a negative amount on the cash account, so the
    amount is negated: a buy adds to invested_eur, a sale subtracts from it.
    The last row is always the account's latest booking day, so readers can
    tell how current the export is.
    """
    invested: dict[str, float] = defaultdict(float)
    income: dict[str, float] = defaultdict(float)
    last_day = ""
    for row in rows:
        if row.get("Konto", "").strip() != account:
            continue
        day = datetime.strptime(row["Buchungsdatum"].strip(), "%d.%m.%Y").date().isoformat()
        last_day = max(last_day, day)
        category = row.get("Kategorie", "").strip()
        if category not in (INVESTED_CATEGORY, INCOME_CATEGORY):
            continue
        amount = float(row["Betrag"])
        if category == INVESTED_CATEGORY:
            invested[day] -= amount
        else:
            income[day] += amount
    # A zero row on the export's last booking day marks how far the data
    # reaches, even when nothing was invested in the final weeks.
    days = sorted(set(invested) | set(income) | ({last_day} if last_day else set()))
    return [
        {"date": d, "invested_eur": round(invested[d], 2), "income_eur": round(income[d], 2)}
        for d in days
    ]


def convert(input_path: str, output_path: str = "cashflows.csv",
            account: str = DEFAULT_ACCOUNT) -> int:
    """Write cashflows.csv; returns the number of days written."""
    path = Path(input_path)
    if not path.exists():
        print(f"Error: '{input_path}' not found.")
        sys.exit(1)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    flows = daily_cashflows(rows, account)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(flows)
    return len(flows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce a Finanzfluss transactions export to daily portfolio cash flows"
    )
    parser.add_argument("--input", required=True, help="Finanzfluss transactions export CSV")
    parser.add_argument("--output", default="cashflows.csv", help="Output path (default: cashflows.csv)")
    parser.add_argument("--account", default=DEFAULT_ACCOUNT,
                        help=f'Cash account to read (default: "{DEFAULT_ACCOUNT}")')
    args = parser.parse_args()
    days = convert(args.input, args.output, args.account)
    print(f"Wrote {days} days of cash flows to '{args.output}'.")


if __name__ == "__main__":
    main()
