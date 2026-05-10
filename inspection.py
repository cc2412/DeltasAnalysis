from pathlib import Path

import pandas as pd
from variables import ROOT_DIR, DELTA_DIRS


def inspect_parquet(root_dir: str = ROOT_DIR, n_rows: int = 3) -> None:
    """
    Print the columns and a few rows from the first parquet file found under
    each delta subfolder of the first run. Run this before the full pipeline
    to confirm that WORD_I_COL, WORD_J_COL, and DELTA_COL below are correct
    for your actual files.
    """

    run_dirs = sorted(Path(root_dir).glob("run*"))

    if not run_dirs:
        print(f"No run directories found under {root_dir}")
        return

    first_run = run_dirs[0]
    print(f"Inspecting: {first_run}\n")

    for subdir, metric in DELTA_DIRS.items():
        folder = first_run / subdir
        parquets = sorted(folder.glob("*.parquet"))
        if not parquets:
            print(f"  [{metric}] No parquet files found in {folder}")
            continue
        df = pd.read_parquet(parquets[0])
        print(f"  [{metric}] file: {parquets[0].name}")
        print(f"           columns: {df.columns.tolist()}")
        print(f"           dtypes:  {df.dtypes.tolist()}")
        print(df.head(n_rows).to_string(index=False))
        print()

if __name__ == "__main__":
    inspect_parquet()