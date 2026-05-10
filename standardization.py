import pandas as pd
from variables import OUTCOME, PREDICTORS


def standardize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Z-score all predictors and the outcome across the full pooled dataset:
        X_z = (X - mean(X)) / std(X)

    Scaling is computed on the full pooled dataset (not per-run), so that
    coefficients are comparable across runs in the mixed-effects model.
    The raw (unstandardized) columns are kept for diagnostics.

    The scaling parameters (mean, SD) are stored in scale_info so they can
    be reported in the methods section and used to back-transform if needed.
    """
    df = df.copy()
    scale_info: dict[str, dict[str, float]] = {}

    cols_to_scale = [OUTCOME] + PREDICTORS
    for col in cols_to_scale:
        mu = df[col].mean()
        sigma = df[col].std(ddof=1)  # ddof=1: sample SD, consistent with standard practice
        if sigma == 0:
            raise ValueError(
                f"Column '{col}' has zero variance -- cannot standardize. "
                f"Check that the parquet files contain real delta values."
            )
        df[f"{col}_z"] = (df[col] - mu) / sigma
        scale_info[col] = {"mean": float(mu), "sd": float(sigma)}
        print(f"  {col}: mean={mu:.5f}, sd={sigma:.5f}")

    return df, scale_info
