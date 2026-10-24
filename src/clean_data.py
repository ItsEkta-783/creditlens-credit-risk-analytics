"""Load and clean the supplied UCI credit-card default workbook."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "default of credit card clients.xls"
OUTPUT_FILE = ROOT / "data" / "processed" / "credit_default_clean.csv"
TARGET = "default_payment_next_month"


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read the .xls sheet; row 1 is metadata and row 2 contains field names."""
    if not path.exists():
        raise FileNotFoundError(f"Expected source workbook at {path}")
    frame = pd.read_excel(path, sheet_name="Data", header=1, engine="xlrd")
    frame.columns = [str(name).strip() for name in frame.columns]
    if "ID" not in frame.columns or "default payment next month" not in frame.columns:
        raise ValueError("Workbook layout differs from the expected credit default dataset")
    return frame


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers and remove invalid or duplicate records, preserving ID."""
    data = frame.copy()
    data.columns = [
        str(name).strip().lower().replace(" ", "_").replace("-", "_")
        for name in data.columns
    ]
    data = data.rename(columns={"pay_0": "pay_1"})
    data = data.drop_duplicates()
    data = data.dropna(subset=[TARGET])
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data[TARGET] = data[TARGET].astype("int8")
    return data.reset_index(drop=True)


def main() -> None:
    data = clean(load_raw())
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(data):,} cleaned rows and {len(data.columns)} columns to {OUTPUT_FILE.relative_to(ROOT)}")
    print(f"Removed {len(load_raw()) - len(data):,} rows with duplicate or unusable target values.")


if __name__ == "__main__":
    main()
