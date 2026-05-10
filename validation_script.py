from pathlib import Path

import pandas as pd

from validation import validate

# Path to the combined parquet file
PARQUET_PATH = Path("processed_data/all_runs_merged.parquet")

# Load dataframe
df = pd.read_parquet(PARQUET_PATH)

print(f"Loaded dataframe with {len(df):,} rows")
print(df.head())

# Run validation
validate(df)

print("\nValidation complete.")