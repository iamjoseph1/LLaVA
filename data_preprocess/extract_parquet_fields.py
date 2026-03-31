#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("../file-000.parquet")
OUTPUT_PATH = Path("extracted_parquet.csv")
REQUIRED_COLUMNS = ["task_category", "language_instruction"]
DROP_NA = True
DROP_DUPLICATES = False


def read_required_columns(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return pd.read_parquet(input_path, columns=REQUIRED_COLUMNS)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if DROP_NA:
        cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    if DROP_DUPLICATES:
        cleaned = cleaned.drop_duplicates(subset=REQUIRED_COLUMNS)
    return cleaned.reset_index(drop=True)


def main() -> None:
    extracted = read_required_columns(INPUT_PATH)
    extracted = clean_dataframe(extracted)
    print(f"Input path: {INPUT_PATH}")
    print(f"Row count: {len(extracted)}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    extracted.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved CSV to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
