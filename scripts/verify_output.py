import pandas as pd
import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE_ROOT / "data" / "processed" / "audio_features_output"

# Load parquet
pq_path = OUTPUT_DIR / "audio_features.parquet"
df = pd.read_parquet(pq_path)

print(f"Parquet shape: {df.shape}")
print(f"Unique chickens: {df['chicken_id'].nunique()}")
print(f"\nFirst 5 rows (chicken_id, duration_sec):")
print(df[['chicken_id', 'duration_sec']].head())

print(f"\nColumns in parquet: {len(df.columns)}")
print(f"Sample columns: {list(df.columns)[:15]}")

print(f"\nData types sample:")
print(df.dtypes.head(10))

# Check if R29-07 is in there
if 'R29-07' in df['chicken_id'].values:
    print("\n[!] R29-07 found in dataframe")
else:
    print("\n[!] R29-07 NOT found in dataframe (expected - it failed during extraction)")

# Load CSV and compare
csv_path = OUTPUT_DIR / "audio_features.csv"
df_csv = pd.read_csv(csv_path)
print(f"\nCSV shape: {df_csv.shape}")
print(f"Parquet == CSV? {df.shape == df_csv.shape}")

print("\n=== SUMMARY ===")
print(f"Total rows: {len(df)}")
print(f"Expected: 246 (minus 1 failed) = 245")
print(f"Match: {len(df) == 245}")
