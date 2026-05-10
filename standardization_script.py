from pathlib import Path
import pandas as pd
import json

from standardization import standardize


# Paths
INPUT_PARQUET = Path("processed_data/all_runs_merged.parquet")

OUTPUT_PARQUET = Path("processed_data/all_runs_merged_standardized.parquet")
OUTPUT_SCALEINFO = Path("processed_data/scale_info.json")

# Load dataframe
df = pd.read_parquet(INPUT_PARQUET)
print(f"Loaded dataframe with {len(df):,} rows")

# Standardize
df_standardized, scale_info = standardize(df)

# Save outputs
df_standardized.to_parquet(OUTPUT_PARQUET, index=False)

with open(OUTPUT_SCALEINFO, "w") as f:
    json.dump(scale_info, f, indent=2)

print(f"\nSaved standardized dataframe to:")
print(f"  {OUTPUT_PARQUET}")

print(f"\nSaved scaling information to:")
print(f"  {OUTPUT_SCALEINFO}")