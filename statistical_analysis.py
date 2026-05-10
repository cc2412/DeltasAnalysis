import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import os
import seaborn as sns
#from variables import *
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pygam import LinearGAM, s
from statsmodels.stats.outliers_influence import variance_inflation_factor

PREDICTORS = ["delta_freq_deltas_z", "delta_conc_deltas_z", "delta_phon_deltas_z"]
OUTCOME = "delta_aoa_deltas_z"

# ---------------------------------------------------------------------------
# Step 4: Exploratory Statistics (Spearman Correlations)
# ---------------------------------------------------------------------------

def exploratory_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman correlations between all variables, per run and overall.

    Spearman is preferred over Pearson here because:
    - Embedding distances are not guaranteed to be normally distributed
    - Relationships may be monotonic but not strictly linear
    - It is robust to outliers in distance distributions

    Eva's suggestion: report mean correlation and standard error across runs.
    """
    all_cols = [OUTCOME] + PREDICTORS
    run_corr_records = []

    for run_id, grp in df.groupby("run"):
        for col in PREDICTORS:
            rho, pval = stats.spearmanr(grp[OUTCOME], grp[col])
            run_corr_records.append({
                "run": run_id,
                "predictor": col,
                "spearman_rho": rho,
                "p_value": pval,
            })

    corr_df = pd.DataFrame(run_corr_records)

    # Summary across runs
    summary = (
        corr_df.groupby("predictor")["spearman_rho"]
        .agg(mean_rho="mean", se_rho=lambda x: x.std() / np.sqrt(len(x)))
        .reset_index()
    )

    print("\n--- Spearman Correlations: Mean across runs ---")
    print(summary.to_string(index=False))

    # Full correlation matrix on pooled data
    full_matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)
    for c1 in all_cols:
        for c2 in all_cols:
            rho, _ = stats.spearmanr(df[c1], df[c2])
            full_matrix.loc[c1, c2] = rho

    return corr_df, summary, full_matrix


def plot_correlation_matrix(full_matrix: pd.DataFrame, output_dir: str):
    """Heatmap of the pooled Spearman correlation matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        full_matrix.astype(float),
        annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, ax=ax,
        square=True, linewidths=0.5,
    )
    ax.set_title("Spearman Correlation Matrix (pooled)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print("  Saved: correlation_matrix.png")


# ---------------------------------------------------------------------------
# Step 5: Per-Run OLS Linear Regression
# ---------------------------------------------------------------------------

def per_run_ols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a separate OLS regression for each run:
        delta_aoa_deltas_z ~ delta_freq_deltas_z + delta_conc_deltas_z + delta_phon_deltas_z

    This establishes:
    - Direction of each predictor effect
    - Effect size (standardized beta)
    - Within-run significance

    Reporting beta stability across runs is a key result (see Eva's comments
    on stability and the table of mean beta / SE across runs).
    """
    records = []
    formula = "delta_aoa_deltas_z ~ delta_freq_deltas_z + delta_conc_deltas_z + delta_phon_deltas_z"

    for run_id, grp in df.groupby("run"):
        model = smf.ols(formula, data=grp).fit()
        for term in PREDICTORS:
            zterm = f"{term}_z"
            records.append({
                "run": run_id,
                "predictor": term,
                "beta": model.params.get(zterm, np.nan),
                "ci_low": model.conf_int().loc[zterm, 0] if zterm in model.conf_int().index else np.nan,
                "ci_high": model.conf_int().loc[zterm, 1] if zterm in model.conf_int().index else np.nan,
                "p_value": model.pvalues.get(zterm, np.nan),
                "r2": model.rsquared,
                "r2_adj": model.rsquared_adj,
                "n": int(model.nobs),
            })

    ols_df = pd.DataFrame(records)
    return ols_df


def summarize_per_run_ols(ols_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-run coefficients: mean beta and SE across runs.
    This is the stability-across-runs table that Eva discussed.
    """
    summary = (
        ols_df.groupby("predictor")
        .agg(
            mean_beta=("beta", "mean"),
            se_beta=("beta", lambda x: x.std() / np.sqrt(len(x))),
            mean_r2=("r2", "mean"),
        )
        .reset_index()
    )
    print("\n--- Per-Run OLS: Stability Summary ---")
    print(summary.to_string(index=False))
    return summary


def plot_coefficient_stability(ols_df: pd.DataFrame, output_dir: str):
    """
    Plot per-run beta estimates with confidence intervals for each predictor.
    This directly visualises coefficient stability across runs.
    """
    predictors = ols_df["predictor"].unique()
    fig, axes = plt.subplots(1, len(predictors), figsize=(5 * len(predictors), 4), sharey=False)

    if len(predictors) == 1:
        axes = [axes]

    for ax, pred in zip(axes, predictors):
        sub = ols_df[ols_df["predictor"] == pred].sort_values("run")
        ax.errorbar(
            sub["run"], sub["beta"],
            yerr=[sub["beta"] - sub["ci_low"], sub["ci_high"] - sub["beta"]],
            fmt="o", capsize=4, color="steelblue",
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Run")
        ax.set_ylabel("Standardized Beta")
        ax.set_title(pred)
        ax.set_xticks(sub["run"].values)

    fig.suptitle("Per-Run OLS: Coefficient Stability", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ols_coefficient_stability.png"), dpi=150)
    plt.close(fig)
    print("  Saved: ols_coefficient_stability.png")


# ---------------------------------------------------------------------------
# Step 6: Mixed-Effects Model (Primary Inferential Model)
# ---------------------------------------------------------------------------

def mixed_effects_model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model pooling all runs:

        delta_aoa_deltas_z ~ delta_freq_deltas_z + delta_conc_deltas_z + delta_phon_deltas_z + (1 | run)

    The random intercept for 'run' accounts for the fact that each run samples
    a different set of word pairs. This is the recommended main analysis because:
    - It uses all data jointly (more power)
    - It explicitly models run-level sampling variability
    - Coefficients generalise across the run sampling distribution

    In statsmodels MixedLM:
        groups = run  <=>  random intercept by run
    """
    formula = "delta_aoa_deltas_z ~ delta_freq_deltas_z + delta_conc_deltas_z + delta_phon_deltas_z"
    model = smf.mixedlm(formula, df, groups=df["run"])
    result = model.fit(reml=True)  # REML is standard for variance estimation

    print("\n--- Mixed-Effects Model Summary ---")
    print(result.summary())
    return result


def extract_mixed_effects_table(result) -> pd.DataFrame:
    """Extract fixed-effect estimates, CIs, and p-values into a clean dataframe."""
    fe = result.fe_params
    ci = result.conf_int()
    pv = result.pvalues

    records = []
    for term in fe.index:
        records.append({
            "term": term,
            "beta": fe[term],
            "ci_low": ci.loc[term, 0],
            "ci_high": ci.loc[term, 1],
            "p_value": pv[term],
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 7: Interaction Models
# ---------------------------------------------------------------------------

def interaction_models(df: pd.DataFrame) -> dict:
    """
    Test two theoretically motivated interaction models:

    1. Frequency x Concreteness interaction:
       Motivation: frequency effects on AoA may be stronger for concrete words,
       which are learned earlier and more grounded in perceptual experience.

    2. Frequency x Phonological interaction:
       Motivation: phonological complexity may matter more at extremes of
       frequency, where other acquisition cues are absent.

    Model comparison uses AIC/BIC and likelihood-ratio tests (LRT).
    LRT is appropriate for nested models fit by maximum likelihood (not REML).

    Note: do not test all possible interactions; only theory-motivated ones
    to avoid multiple comparison inflation.
    """
    base_formula = "delta_aoa_deltas_z ~ delta_freq_deltas_z + delta_conc_deltas_z + delta_phon_deltas_z"
    int1_formula = "delta_aoa_z ~ delta_freq_z * delta_conc_z + delta_phon_z"
    int2_formula = "delta_aoa_z ~ delta_freq_z * delta_phon_z + delta_conc_z"

    results = {}
    for name, formula in [("additive", base_formula),
                          ("freq_x_conc", int1_formula),
                          ("freq_x_phon", int2_formula)]:
        # Use ML (not REML) for likelihood-ratio tests between fixed effects
        m = smf.mixedlm(formula, df, groups=df["run"]).fit(reml=False)
        results[name] = m
        print(f"\n  Model '{name}': AIC={m.aic:.2f}, BIC={m.bic:.2f}, loglik={m.llf:.2f}")

    # Likelihood-ratio test: additive vs. interaction models
    for name in ["freq_x_conc", "freq_x_phon"]:
        lr_stat = 2 * (results[name].llf - results["additive"].llf)
        # Each interaction adds 1 degree of freedom
        df_diff = 1
        p_lrt = stats.chi2.sf(lr_stat, df=df_diff)
        print(f"\n  LRT additive vs {name}: chi2={lr_stat:.3f}, df={df_diff}, p={p_lrt:.4f}")

    return results


# ---------------------------------------------------------------------------
# Step 8: GAM (Generalized Additive Model)
# ---------------------------------------------------------------------------

def fit_gam(df: pd.DataFrame):
    """
    Fit a GAM to test whether predictor-outcome relationships are nonlinear:

        delta_aoa_z ~ s(delta_freq_z) + s(delta_conc_z) + s(delta_phon_z)

    where s(.) are penalized spline smooth functions estimated from the data.

    Theoretical motivation (from meeting notes): the loss geometry of embedding
    training has flat regions and steep areas. This could induce nonlinear
    mappings between lexical property distances. The GAM detects such structure
    without assuming a functional form.

    Role: robustness check and exploratory analysis, not the primary confirmatory
    model. If the GAM smooth for a predictor is approximately linear, the OLS
    assumption is validated. If it is nonlinear, that is an important finding
    for the discussion.
    """
    X = df[["delta_freq_z", "delta_conc_z", "delta_phon_z"]].values
    y = df["delta_aoa_z"].values

    gam = LinearGAM(
        s(0) + s(1) + s(2),
        max_iter=100,
    ).gridsearch(X, y)

    print("\n--- GAM Summary ---")
    gam.summary()
    return gam


def plot_gam_partial_effects(gam, feature_names: list, output_dir: str):
    """
    Plot partial response curves for each smooth term in the GAM.
    These show the marginal effect of each predictor on AoA delta,
    holding other predictors at their mean.
    """
    fig, axes = plt.subplots(1, len(feature_names), figsize=(5 * len(feature_names), 4))

    if len(feature_names) == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        ax.plot(XX[:, i], pdep, color="steelblue", linewidth=2)
        ax.fill_between(XX[:, i], confi[:, 0], confi[:, 1], alpha=0.25, color="steelblue")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(name)
        ax.set_ylabel("Partial effect on delta_aoa_z")
        ax.set_title(f"GAM smooth: {name}")

    fig.suptitle("GAM Partial Effects", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gam_partial_effects.png"), dpi=150)
    plt.close(fig)
    print("  Saved: gam_partial_effects.png")


# ---------------------------------------------------------------------------
# Step 9: Model Diagnostics
# ---------------------------------------------------------------------------

def plot_diagnostics(df: pd.DataFrame, result, output_dir: str):
    """
    Standard OLS regression diagnostics applied to the pooled additive model.
    Key checks:
    - Residuals vs. fitted: should show no systematic pattern (homoskedasticity)
    - QQ plot: checks normality of residuals
    - Leverage plot: identifies high-influence observations

    Violations:
    - Heteroskedasticity: consider robust (HC3) standard errors
    - Non-normality: consider bootstrap CIs or quantile regression
    """
    # Fit a plain OLS on pooled data for diagnostics (mixed model residuals are similar)
    formula = "delta_aoa_deltas_z ~ delta_freq_deltas_z + delta_conc_deltas_z + delta_phon_deltas_z"
    ols_result = smf.ols(formula, data=df).fit()

    fitted = ols_result.fittedvalues
    residuals = ols_result.resid

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Residuals vs. fitted
    axes[0].scatter(fitted, residuals, alpha=0.1, s=5, color="steelblue")
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs. Fitted")

    # QQ plot
    sm.qqplot(residuals, line="s", ax=axes[1], alpha=0.3, markersize=2)
    axes[1].set_title("QQ Plot of Residuals")

    # Scale-location (sqrt of absolute residuals vs fitted)
    axes[2].scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.1, s=5, color="steelblue")
    axes[2].set_xlabel("Fitted values")
    axes[2].set_ylabel("sqrt(|Residuals|)")
    axes[2].set_title("Scale-Location")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "diagnostics.png"), dpi=150)
    plt.close(fig)
    print("  Saved: diagnostics.png")

    # VIF: check for multicollinearity among predictors
    X_cols = ["delta_freq_z", "delta_conc_z", "delta_phon_z"]
    X_vif = df[X_cols].copy()
    X_vif = sm.add_constant(X_vif)
    vif_data = pd.DataFrame({
        "feature": X_cols,
        "VIF": [variance_inflation_factor(X_vif.values, i + 1) for i in range(len(X_cols))]
    })
    print("\n--- Variance Inflation Factors (multicollinearity check) ---")
    print(vif_data.to_string(index=False))
    return vif_data


# ---------------------------------------------------------------------------
# Step 10: Coefficient Plot (summary figure)
# ---------------------------------------------------------------------------

def plot_coefficient_summary(mixed_table: pd.DataFrame, ols_stability: pd.DataFrame,
                             output_dir: str):
    """
    Forest-style plot of mixed-effects fixed effects with 95% CIs.
    This is the primary summary figure showing direction, size, and uncertainty
    of each predictor's effect on AoA delta.
    """
    sub = mixed_table[mixed_table["term"].str.startswith("delta_")].copy()
    sub["label"] = sub["term"].str.replace("_z", "").str.replace("delta_", "")

    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = range(len(sub))

    ax.errorbar(
        sub["beta"].values,
        list(y_pos),
        xerr=[
            (sub["beta"] - sub["ci_low"]).values,
            (sub["ci_high"] - sub["beta"]).values,
        ],
        fmt="o", color="steelblue", capsize=5, linewidth=2, markersize=7,
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sub["label"].values, fontsize=11)
    ax.set_xlabel("Standardized Beta (95% CI)", fontsize=11)
    ax.set_title("Mixed-Effects Model: Fixed Effects on delta_AoA", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mixed_effects_coefficients.png"), dpi=150)
    plt.close(fig)
    print("  Saved: mixed_effects_coefficients.png")


# ---------------------------------------------------------------------------
# Step 11: Save Tables
# ---------------------------------------------------------------------------

def save_tables(corr_summary, ols_stability, mixed_table, vif_data, output_dir):
    """Save all key result tables as CSV for inclusion in the paper."""
    corr_summary.to_csv(os.path.join(output_dir, "table_spearman_correlations.csv"), index=False)
    ols_stability.to_csv(os.path.join(output_dir, "table_ols_stability.csv"), index=False)
    mixed_table.to_csv(os.path.join(output_dir, "table_mixed_effects.csv"), index=False)
    vif_data.to_csv(os.path.join(output_dir, "table_vif.csv"), index=False)
    print("\n  Saved all result tables to CSV.")
