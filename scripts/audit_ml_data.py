"""Audit committed fictional CSV headers for Chapter 21's limited field guard."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harbor_ml.data_security import (  # noqa: E402
    find_prohibited_fields,
    validate_dataset_columns,
)


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as source:
        return next(csv.reader(source), [])


def main() -> int:
    print("Harbor Federal Credit Union\nML Dataset Header Audit\n")
    failed = False
    for path in sorted((ROOT / "data").glob("*.csv")):
        columns = read_header(path)
        problems = list(find_prohibited_fields(columns))
        if path.name == "harbor_integration_requests.csv":
            try:
                validate_dataset_columns(columns)
            except ValueError as error:
                problems.append(str(error))
        status = "FAIL: " + "; ".join(problems) if problems else "PASS"
        failed |= bool(problems)
        print(f"{path.name:<43} {status}")
        print(f"  columns: {', '.join(columns)}")
    print("\nLimited guard: exact header names only; values are not inspected.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
