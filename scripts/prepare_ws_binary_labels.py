#!/usr/bin/env python3
"""
Prepare a binary White Striping label table from PolliMIOPATIE.xlsx.

The source workbook contains ordinal WS grades. This script converts them to a
binary target where 0 stays 0 and any positive grade becomes 1, while keeping
the original grade in a separate traceable column.
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_XLSX = BASE_DIR / "data" / "raw" / "PolliMIOPATIE.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "ws_labels_binary.parquet"
OUTPUT_CSV = OUTPUT_DIR / "ws_labels_binary.csv"


def main() -> None:
    print("=" * 70)
    print("PREPARE WS BINARY LABELS")
    print("=" * 70)

    if not RAW_XLSX.exists():
        raise FileNotFoundError(f"Source workbook not found: {RAW_XLSX}")

    df = pd.read_excel(RAW_XLSX)

    required_columns = {"Matricola", "Grado WS"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in workbook: {sorted(missing_columns)}")

    labels = df[["Matricola", "Grado WS"]].copy()
    labels["chicken_id"] = labels["Matricola"].astype(str).str.strip().str.replace("/", "-", regex=False)
    labels["ws_grade_raw"] = pd.to_numeric(labels["Grado WS"], errors="coerce")

    valid_grades = {0.0, 0.5, 1.0, 1.5, 2.0}
    unexpected_values = sorted(set(labels["ws_grade_raw"].dropna().unique()) - valid_grades)
    if unexpected_values:
        raise ValueError(f"Unexpected values in 'Grado WS': {unexpected_values}")

    labels["ws_myopathy_binary"] = labels["ws_grade_raw"].map(lambda value: 0 if value == 0 else 1)
    labels = labels[["chicken_id", "ws_grade_raw", "ws_myopathy_binary"]].drop_duplicates("chicken_id")

    if labels["ws_grade_raw"].isna().any():
        raise ValueError("Some 'Grado WS' values could not be converted to numeric.")

    labels.to_parquet(OUTPUT_PARQUET, index=False)
    labels.to_csv(OUTPUT_CSV, index=False)

    print(f"✓ Saved parquet: {OUTPUT_PARQUET}")
    print(f"✓ Saved csv: {OUTPUT_CSV}")
    print(f"✓ Rows: {len(labels)}")
    print(f"✓ Unique binary labels: {labels['ws_myopathy_binary'].value_counts().to_dict()}")
    print(f"✓ Raw WS grades: {sorted(labels['ws_grade_raw'].unique().tolist())}")


if __name__ == "__main__":
    main()