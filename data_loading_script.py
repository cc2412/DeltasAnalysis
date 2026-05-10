from pathlib import Path
import pandas as pd

# Import your loader
from data_loading import load_all_runs

# Output location
OUTPUT_DIR = Path("processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Build pooled dataframe
df = load_all_runs()

# Optional but strongly recommended memory optimization
for col in ["word_i", "word_j", "run"]:
    df[col] = df[col].astype("category")

# Save as parquet (recommended main format)
parquet_path = OUTPUT_DIR / "all_runs_merged.parquet"
df.to_parquet(parquet_path, index=False)

# Optional CSV export for inspection/debugging
csv_path = OUTPUT_DIR / "all_runs_merged.csv"
df.to_csv(csv_path, index=False)

print("\nSaved files:")
print(f"  Parquet: {parquet_path.resolve()}")
print(f"  CSV:     {csv_path.resolve()}")

print("\nFinal dataframe info:")
print(df.info())

print("\nFirst few rows:")
print(df.head())