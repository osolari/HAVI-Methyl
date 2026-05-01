"""Shared CLI plumbing for figure / table / benchmark scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure ``src`` is on the path when the scripts are invoked directly.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--seed", type=int, default=20260429, help="Random seed (default: 20260429 — paper)."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run with smaller S/L/iter for quick iteration.",
    )
    return parser


def small_or_full(fast: bool, full: tuple, fast_values: tuple) -> tuple:
    """Return either ``full`` or ``fast_values`` based on ``--fast``."""
    return fast_values if fast else full


def write_json(path: str | Path, data) -> Path:
    """Write JSON, ensuring the parent directory exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(data, f, indent=2)
    return p


def write_csv(path: str | Path, rows: list[dict]) -> Path:
    """Write a CSV from a list of dicts; keys of the first row determine columns."""
    import csv as _csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return p
    keys = list(rows[0].keys())
    with p.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return p


def copy_to_report_tables(path: Path) -> Path:
    """Copy a generated CSV into ``docs/report/tables`` (used by LaTeX builds)."""
    target = Path("docs/report/tables") / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    return target
