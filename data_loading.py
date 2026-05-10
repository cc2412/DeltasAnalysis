import re
from pathlib import Path

import pandas as pd

from variables import DELTA_DIRS, WORD_I_COL, WORD_J_COL, DELTA_COL, ROOT_DIR


def load_metric_for_run(run_dir: Path, metric: str) -> pd.DataFrame:
    """
    Load all training steps for one metric in one run and concatenate them.
    Each tranche parquet file corresponds to one training step.
    We add a 'training_step' column derived from the filename so that, after merging across
    metrics, each row uniquely identifies a (word_i, word_j, training_step) triple.
    The delta column is renamed to delta_<metric> to avoid ambiguity.

    Returns a dataframe with columns:
        word_i, word_j, training_step, delta_<metric>
    """

    # Find all .parquet files in a folder and sort them to ensure
    # that files are processed in a deterministic order.
    folder = run_dir / metric
    parquet_files = sorted(folder.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError("No parquet files found in {}".format(folder))

    frames: list[pd.DataFrame] = []

    for f in parquet_files:
        df = pd.read_parquet(f)
        match = re.search(r"(\d+)", f.stem)
        step_label = int(match.group(1))
        df = df[[WORD_I_COL, WORD_J_COL, DELTA_COL]].copy()
        df = df.rename(columns={
            WORD_I_COL: "word_i",
            WORD_J_COL: "word_j",
            DELTA_COL: f"delta_{metric}",
        })
        df["training_step"] = step_label
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def merge_metrics_for_run(run_dir: Path, run_id: str) -> pd.DataFrame:
    """
    Load all four metrics for one run and merge them into a single dataframe.

    Merge on (word_i, word_j, training_step) because different metrics may be
    stored with different row orderings and some word pairs are absent for some
    metrics. Row-wise concatenation would misalign pairs.

    Returns a dataframe with columns:
        word_i, word_j, run, training_step, delta_aoa, delta_freq, delta_phon, delta_conc
    """
    merge_key = ["word_i", "word_j", "training_step"]

    # Load each metric independently
    metric_frames: dict[str, pd.DataFrame] = {}
    for metric in DELTA_DIRS:
        mdf = load_metric_for_run(run_dir, metric)
        metric_frames[metric] = mdf
        print(f"    [{metric}] {len(mdf):>7,} rows across all tranches")

    merged = metric_frames["aoa"].copy()
    n_before = len(merged)

    for metric in ("freq", "phon", "conc"):
        n_before_this = len(merged)
        merged = merged.merge(
            metric_frames[metric],
            on=merge_key,
            how="inner",
            validate="one_to_one",
        )
        n_dropped = n_before_this - len(merged)
        if n_dropped > 0:
            print(
                f"    inner join with '{metric}': dropped {n_dropped:,} rows "
                f"({100 * n_dropped / n_before_this:.1f}%) -- pairs absent from '{metric}'"
            )

    # Attach run identifier.
    merged["run"] = run_id

    total_dropped = n_before - len(merged)
    print(
        f"    Run {run_id}: {n_before:,} AoA rows -> "
        f"{len(merged):,} complete rows after all inner joins "
        f"({total_dropped:,} dropped, {100 * total_dropped / max(n_before, 1):.1f}%)"
    )

    return merged

def load_all_runs(root_dir: Path = Path(ROOT_DIR)) -> pd.DataFrame:
    """
    Discover all run directories, load and merge each, then pool into one dataframe.
    """
    # Accept both "run 0" and "run_0" style names.
    run_dirs = sorted(root_dir.glob("run*"))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {root_dir}")

    all_frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        # Parse a clean run id from the folder name (last token of digits).
        folder_name = run_dir.name
        digits = "".join(ch for ch in folder_name if ch.isdigit())
        run_id = int(digits) if digits else folder_name
        print(f"\n  Run {run_id}  ({run_dir})")
        run_df = merge_metrics_for_run(run_dir, run_id)
        all_frames.append(run_df)

    pooled = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal complete observations pooled across all runs: {len(pooled):,}")

    return pooled