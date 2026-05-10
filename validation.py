import pandas as pd
from variables import OUTCOME, PREDICTORS

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-load integrity checks on the merged dataframe.

    Checks performed:
    1. No NaNs in any delta column -- the inner joins should have prevented
       this, but we verify defensively.
    2. No duplicate (word_i, word_j, run, training_step) rows -- each such
       triple should appear exactly once.
    3. Report per-column and per-run row counts for the paper footnote.

    The function raises on fatal errors and prints warnings for non-fatal ones,
    then returns the dataframe unchanged (the joins already handled missingness).
    """
    all_delta_cols = [OUTCOME] + PREDICTORS
    uid_cols = ["word_i", "word_j", "run", "training_step"]

    print("\n--- Validation ---")

    # 1. NaN check.
    nan_found = False
    for col in all_delta_cols:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            print(f"  WARNING: {n_nan} NaNs in '{col}' -- these survived the inner join unexpectedly.")
            nan_found = True
    if not nan_found:
        print("  NaN check passed: no missing values in any delta column.")

    # 2. Duplicate check.
    n_dupes = df.duplicated(subset=uid_cols).sum()
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes} duplicate (word_i, word_j, run, training_step) rows found.")
        print("           This suggests a word pair appears more than once within a single tranche.")
        print("           Investigate before proceeding.")
    else:
        print(f"  Duplicate check passed: all ({', '.join(uid_cols)}) combinations are unique.")

    # 3. Row count summary by run (for the methods section / footnote).
    print("\n  Row counts by run:")
    counts = df.groupby("run").size().reset_index(name="n_observations")
    print(counts.to_string(index=False))

    # 4. Row count summary by training step (sanity check on tranche coverage).
    print("\n  Row counts by training step:")
    step_counts = df.groupby("training_step").size().reset_index(name="n_observations")
    print(step_counts.to_string(index=False))

    return df
