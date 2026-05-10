from pathlib import Path

import pandas as pd

from statistical_analysis import (
    exploratory_correlations,
    plot_correlation_matrix,
    per_run_ols,
    summarize_per_run_ols,
    plot_coefficient_stability,
    mixed_effects_model,
    extract_mixed_effects_table,
    interaction_models,
    fit_gam,
    plot_gam_partial_effects,
    plot_diagnostics,
    plot_coefficient_summary,
    save_tables,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

DATA_PATH = Path("processed_data/all_runs_merged_standardized.parquet")
OUTPUT_DIR = Path("analysis_outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load standardized dataset
# ------------------------------------------------------------------

print("Loading standardized parquet...")
df = pd.read_parquet(DATA_PATH)

print(f"Loaded dataframe with {len(df):,} rows")
print(df.head())

df.columns = df.columns.str.replace(" ", "_")


# ------------------------------------------------------------------
# Step 4: Exploratory correlations
# ------------------------------------------------------------------

print("\n==============================")
print("STEP 4: EXPLORATORY CORRELATIONS")
print("==============================")

corr_df, corr_summary, full_matrix = exploratory_correlations(df)

plot_correlation_matrix(
    full_matrix=full_matrix,
    output_dir=str(OUTPUT_DIR),
)

# ------------------------------------------------------------------
# Step 5: Per-run OLS
# ------------------------------------------------------------------

print("\n==============================")
print("STEP 5: PER-RUN OLS")
print("==============================")

ols_df = per_run_ols(df)

ols_stability = summarize_per_run_ols(ols_df)

plot_coefficient_stability(
    ols_df=ols_df,
    output_dir=str(OUTPUT_DIR),
)

# ------------------------------------------------------------------
# Step 6: Mixed-effects model
# ------------------------------------------------------------------

print("\n==============================")
print("STEP 6: MIXED-EFFECTS MODEL")
print("==============================")

mixed_result = mixed_effects_model(df)

mixed_table = extract_mixed_effects_table(mixed_result)

print("\nMixed-effects fixed effects:")
print(mixed_table.to_string(index=False))

# ------------------------------------------------------------------
# Step 7: Interaction models
# ------------------------------------------------------------------

print("\n==============================")
print("STEP 7: INTERACTION MODELS")
print("==============================")

interaction_results = interaction_models(df)

# ------------------------------------------------------------------
# Step 8: GAM
# ------------------------------------------------------------------

print("\n==============================")
print("STEP 8: GAM")
print("==============================")

gam = fit_gam(df)

plot_gam_partial_effects(
    gam=gam,
    feature_names=[
        "delta_freq_z",
        "delta_conc_z",
        "delta_phon_z",
    ],
    output_dir=str(OUTPUT_DIR),
)

# ------------------------------------------------------------------
# Step 9: Diagnostics
# ------------------------------------------------------------------

print("\n==============================")
print("STEP 9: DIAGNOSTICS")
print("==============================")

vif_data = plot_diagnostics(
    df=df,
    result=mixed_result,
    output_dir=str(OUTPUT_DIR),
)

# ------------------------------------------------------------------
# Step 10: Coefficient summary figure
# ------------------------------------------------------------------

print("\n==============================")
print("STEP 10: COEFFICIENT SUMMARY")
print("==============================")

plot_coefficient_summary(
    mixed_table=mixed_table,
    ols_stability=ols_stability,
    output_dir=str(OUTPUT_DIR),
)

# ------------------------------------------------------------------
# Step 11: Save tables
# ------------------------------------------------------------------

print("\n==============================")
print("STEP 11: SAVE TABLES")
print("==============================")

save_tables(
    corr_summary=corr_summary,
    ols_stability=ols_stability,
    mixed_table=mixed_table,
    vif_data=vif_data,
    output_dir=str(OUTPUT_DIR),
)

# ------------------------------------------------------------------
# Optional: save detailed per-run OLS + correlations
# ------------------------------------------------------------------

corr_df.to_csv(
    OUTPUT_DIR / "table_per_run_correlations.csv",
    index=False,
)

ols_df.to_csv(
    OUTPUT_DIR / "table_per_run_ols.csv",
    index=False,
)

print("\nAnalysis complete.")
print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")