#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm


def split_config_id(config_id: str) -> Tuple[str, float]:
    # expects: T0_temp0.2
    if isinstance(config_id, str) and "_temp" in config_id:
        t, temp = config_id.split("_temp", 1)
        try:
            return t, float(temp)
        except ValueError:
            return t, float("nan")
    return str(config_id), float("nan")


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def summarize_config(df_perq: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce per-question rows to config-level means for plotting.
    """
    g = df_perq.groupby("config_id", dropna=False)
    out = pd.DataFrame(
        {
            "treatment": g["treatment"].first(),
            "temp": g["temp"].first(),
            "n_questions": g["question_id"].nunique(),
            "k_runs": g["k_runs"].max(),
            "accuracy_mean": g["accuracy_over_runs"].mean(),
            "strict_stability_rate": g["strict_stable"].mean(),
            "entropy_mean_bits": g["answer_entropy_bits"].mean(),
            "mode_freq_mean": g["mode_freq"].mean(),
        }
    ).reset_index()
    return out.sort_values(["treatment", "temp"]).reset_index(drop=True)


def fit_binomial_glm(df_perq: pd.DataFrame, interaction: bool):
    """
    Binomial GLM logit with aggregated trials per question:
      successes ~ Binomial(k_runs, p)
      logit(p) = ...
    We implement as GLM with response = successes/k_runs and freq_weights=k_runs.
    """
    df = df_perq.copy()
    df["prop"] = df["successes"] / df["k_runs"]

    # set T0 as reference if present
    if "T0" in set(df["treatment"].astype(str)):
        df["treatment"] = pd.Categorical(df["treatment"])
        cats = list(df["treatment"].cat.categories)
        if "T0" in cats:
            cats = ["T0"] + [c for c in cats if c != "T0"]
            df["treatment"] = df["treatment"].cat.reorder_categories(cats, ordered=False)

    formula = "prop ~ C(treatment) + temp"
    name = "glm_additive"
    if interaction:
        formula = "prop ~ C(treatment) * temp"
        name = "glm_interaction"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["k_runs"].astype(float),
    )
    res = model.fit()
    return name, res


def save_coef_table(res, out_csv: str, out_txt: str) -> None:
    ensure_dir(out_csv)
    ensure_dir(out_txt)

    coef = res.params
    se = res.bse
    p = res.pvalues

    or_ = np.exp(coef)
    ci_low = np.exp(coef - 1.96 * se)
    ci_high = np.exp(coef + 1.96 * se)

    tab = pd.DataFrame(
        {
            "term": coef.index,
            "coef_logit": coef.values,
            "std_err": se.values,
            "p_value": p.values,
            "odds_ratio": or_.values,
            "or_ci_low": ci_low.values,
            "or_ci_high": ci_high.values,
        }
    )
    tab.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(res.summary().as_text())


def plot_tradeoff(summary_cfg: pd.DataFrame, out_png: str) -> None:
    ensure_dir(out_png)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    temps = sorted(summary_cfg["temp"].dropna().unique().tolist())
    for temp in temps:
        sub = summary_cfg[summary_cfg["temp"] == temp]
        ax.scatter(sub["entropy_mean_bits"], sub["accuracy_mean"], label=f"temp={temp}", alpha=0.55)
        for _, r in sub.iterrows():
            ax.annotate(
                str(r["treatment"]),
                (r["entropy_mean_bits"], r["accuracy_mean"]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Mean Answer Entropy (bits)")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Accuracy vs Entropy (by config)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_lines_by_temp(summary_cfg: pd.DataFrame, y: str, title: str, out_png: str) -> None:
    ensure_dir(out_png)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for t in sorted(summary_cfg["treatment"].unique().tolist()):
        sub = summary_cfg[summary_cfg["treatment"] == t].sort_values("temp")
        ax.plot(sub["temp"], sub[y], marker="o", label=str(t))

    ax.set_xlabel("Temperature")
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_predicted_curves(res, df_perq: pd.DataFrame, out_png: str) -> None:
    ensure_dir(out_png)

    treatments = sorted(df_perq["treatment"].unique().tolist())
    temps = np.linspace(df_perq["temp"].min(), df_perq["temp"].max(), 50)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for t in treatments:
        grid = pd.DataFrame({"treatment": [t] * len(temps), "temp": temps})
        pred = res.predict(grid)
        ax.plot(temps, pred, label=str(t))

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Predicted P(correct)")
    ax.set_title("Predicted accuracy vs temperature (GLM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =========================
# Added: explicit logistic regression (run-level)
# =========================

def expand_to_run_level(df_perq: pd.DataFrame) -> pd.DataFrame:
    """
    Expand per-question aggregated results into run-level binary outcomes.

    Example:
      k_runs=3, successes=2 -> rows: [1, 1, 0]

    This lets us fit an explicit logistic regression on individual runs while
    clustering SEs by question_id to account for repeated measurements.
    """
    rows = []
    use_cols = ["config_id", "question_id", "treatment", "temp", "k_runs", "successes"]

    for _, r in df_perq[use_cols].iterrows():
        k = int(r["k_runs"])
        s = int(r["successes"])
        s = max(0, min(s, k))

        outcomes = [1] * s + [0] * (k - s)
        for i, y in enumerate(outcomes):
            rows.append(
                {
                    "config_id": r["config_id"],
                    "question_id": r["question_id"],
                    "treatment": r["treatment"],
                    "temp": float(r["temp"]),
                    "run_idx_within_question": i,
                    "correct": int(y),
                }
            )

    df_runs = pd.DataFrame(rows)

    if "T0" in set(df_runs["treatment"].astype(str)):
        df_runs["treatment"] = pd.Categorical(df_runs["treatment"])
        cats = list(df_runs["treatment"].cat.categories)
        if "T0" in cats:
            cats = ["T0"] + [c for c in cats if c != "T0"]
            df_runs["treatment"] = df_runs["treatment"].cat.reorder_categories(cats, ordered=False)

    return df_runs


def fit_logistic_regression(df_runs: pd.DataFrame, interaction: bool):
    """
    Run-level logistic regression with question-clustered robust SE.
    """
    formula = "correct ~ C(treatment) + temp"
    name = "logit_additive"
    if interaction:
        formula = "correct ~ C(treatment) * temp"
        name = "logit_interaction"

    model = smf.glm(
        formula=formula,
        data=df_runs,
        family=sm.families.Binomial(),
    )

    res = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df_runs["question_id"]},
    )
    return name, res


def save_logit_test_table(res, out_csv: str) -> None:
    ensure_dir(out_csv)
    wt = res.wald_test_terms(skip_single=False)
    table = wt.summary_frame().reset_index().rename(columns={"index": "term"})
    table.to_csv(out_csv, index=False)


def save_predicted_probability_table(res, df_source: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    ensure_dir(out_csv)

    treatments = sorted(pd.Series(df_source["treatment"]).astype(str).unique().tolist())
    temps = sorted(pd.Series(df_source["temp"]).dropna().astype(float).unique().tolist())

    grid_rows = []
    for t in treatments:
        for temp in temps:
            grid_rows.append({"treatment": t, "temp": temp})

    grid = pd.DataFrame(grid_rows)
    grid["predicted_p_correct"] = res.predict(grid)
    grid.to_csv(out_csv, index=False)
    return grid


def plot_logit_predictions_with_observed(
    res,
    summary_cfg: pd.DataFrame,
    df_source: pd.DataFrame,
    out_png: str,
) -> None:
    ensure_dir(out_png)

    treatments = sorted(df_source["treatment"].astype(str).unique().tolist())
    temps_dense = np.linspace(df_source["temp"].min(), df_source["temp"].max(), 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for t in treatments:
        grid = pd.DataFrame({"treatment": [t] * len(temps_dense), "temp": temps_dense})
        pred = res.predict(grid)
        ax.plot(temps_dense, pred, label=f"{t} fitted")

        obs = summary_cfg[summary_cfg["treatment"].astype(str) == str(t)].sort_values("temp")
        ax.scatter(obs["temp"], obs["accuracy_mean"])

    ax.set_xlabel("Temperature")
    ax.set_ylabel("P(correct)")
    ax.set_title("Logistic regression fitted probabilities vs observed accuracy")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =========================
# Added: treatment-focused plots
# =========================

def _ordered_treatments(summary_cfg: pd.DataFrame):
    canonical = ["T0", "T1", "T2", "T3", "T4", "T5"]
    present = [t for t in canonical if t in set(summary_cfg["treatment"].astype(str))]
    if present:
        return present
    return sorted(summary_cfg["treatment"].astype(str).unique().tolist())


def _ordered_temps(summary_cfg: pd.DataFrame):
    temps = summary_cfg["temp"].dropna().astype(float).unique().tolist()
    return sorted(temps)


def plot_metric_by_treatment_grouped_by_temp(
    summary_cfg: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_png: str,
) -> None:
    """
    X-axis: treatment
    Groups/legend: temperature
    Y-axis: metric
    """
    ensure_dir(out_png)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    treatments = _ordered_treatments(summary_cfg)
    temps = _ordered_temps(summary_cfg)

    x = np.arange(len(treatments))
    offsets = np.linspace(-0.25, 0.25, len(temps)) if len(temps) > 1 else [0.0]

    for off, temp in zip(offsets, temps):
        sub = summary_cfg[summary_cfg["temp"].astype(float) == float(temp)].copy()
        sub["treatment"] = sub["treatment"].astype(str)
        y = []
        for t in treatments:
            m = sub[sub["treatment"] == t][metric_col]
            y.append(float(m.iloc[0]) if len(m) else np.nan)

        ax.scatter(x + off, y, label=f"temp={temp}")
        ax.plot(x + off, y)

    ax.set_xticks(x)
    ax.set_xticklabels(treatments)
    ax.set_xlabel("Treatment")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_tradeoff_accuracy_vs_stability(summary_cfg: pd.DataFrame, out_png: str) -> None:
    ensure_dir(out_png)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(summary_cfg["strict_stability_rate"], summary_cfg["accuracy_mean"],alpha=0.55)

    for _, r in summary_cfg.iterrows():
        label = f"{r['treatment']}_t{r['temp']}"
        ax.annotate(
            label,
            (r["strict_stability_rate"], r["accuracy_mean"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Strict Stability Rate")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Accuracy vs Strict Stability (by config)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_heatmap_treatment_temp(
    summary_cfg: pd.DataFrame,
    value_col: str,
    title: str,
    out_png: str,
) -> None:
    ensure_dir(out_png)

    dfp = summary_cfg.copy()
    dfp["treatment"] = dfp["treatment"].astype(str)
    dfp["temp"] = dfp["temp"].astype(float)

    treatments = _ordered_treatments(dfp)
    temps = _ordered_temps(dfp)

    pivot = dfp.pivot_table(index="treatment", columns="temp", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=treatments, columns=temps)
    mat = pivot.values.astype(float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Treatment")

    ax.set_xticks(np.arange(len(temps)))
    ax.set_xticklabels([str(t) for t in temps])
    ax.set_yticks(np.arange(len(treatments)))
    ax.set_yticklabels(treatments)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_treatment_only_summary(summary_cfg: pd.DataFrame, out_csv: str) -> None:
    ensure_dir(out_csv)
    g = summary_cfg.groupby("treatment", dropna=False)
    out = pd.DataFrame(
        {
            "n_configs": g.size(),
            "accuracy_mean_over_temps": g["accuracy_mean"].mean(),
            "accuracy_sd_over_temps": g["accuracy_mean"].std(ddof=1),
            "stability_mean_over_temps": g["strict_stability_rate"].mean(),
            "stability_sd_over_temps": g["strict_stability_rate"].std(ddof=1),
            "entropy_mean_over_temps": g["entropy_mean_bits"].mean(),
            "entropy_sd_over_temps": g["entropy_mean_bits"].std(ddof=1),
        }
    ).reset_index()

    order = _ordered_treatments(summary_cfg)
    out["treatment"] = pd.Categorical(out["treatment"].astype(str), categories=order, ordered=True)
    out = out.sort_values("treatment").reset_index(drop=True)
    out.to_csv(out_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-question-csv", required=True, help="outputs/per_question.csv from your script")
    ap.add_argument("--out-dir", default="outputs/glm_analysis")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.per_question_csv)

    # Parse treatment/temp from config_id
    tt = df["config_id"].apply(lambda x: split_config_id(str(x)))
    df["treatment"] = tt.apply(lambda x: x[0])
    df["temp"] = tt.apply(lambda x: x[1])

    # Numeric cleanup
    for c in ["k_runs", "mode_freq", "answer_entropy_bits", "accuracy_over_runs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["strict_stable"] = (
        df["strict_stable"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "t", "yes", "y"])
        .astype(float)
    )

    # Build binomial counts
    df["successes"] = (df["accuracy_over_runs"] * df["k_runs"]).round().astype(int)
    df["successes"] = df["successes"].clip(lower=0, upper=df["k_runs"].astype(int))
    df = df.dropna(subset=["temp", "k_runs", "successes"])

    # Config-level summary (for plots)
    summary_cfg = summarize_config(df)
    summary_path = os.path.join(out_dir, "summary_by_config.csv")
    summary_cfg.to_csv(summary_path, index=False)

    # Fit GLMs
    name_a, res_a = fit_binomial_glm(df, interaction=False)
    save_coef_table(
        res_a,
        os.path.join(out_dir, f"{name_a}_coef.csv"),
        os.path.join(out_dir, f"{name_a}_summary.txt"),
    )

    name_b, res_b = fit_binomial_glm(df, interaction=True)
    save_coef_table(
        res_b,
        os.path.join(out_dir, f"{name_b}_coef.csv"),
        os.path.join(out_dir, f"{name_b}_summary.txt"),
    )

    # Original plots
    plot_tradeoff(summary_cfg, os.path.join(out_dir, "tradeoff_accuracy_vs_entropy.png"))
    plot_lines_by_temp(
        summary_cfg,
        "accuracy_mean",
        "Accuracy vs Temperature",
        os.path.join(out_dir, "accuracy_vs_temp.png"),
    )
    plot_lines_by_temp(
        summary_cfg,
        "strict_stability_rate",
        "Strict Stability vs Temperature",
        os.path.join(out_dir, "stability_vs_temp.png"),
    )
    plot_lines_by_temp(
        summary_cfg,
        "entropy_mean_bits",
        "Entropy vs Temperature",
        os.path.join(out_dir, "entropy_vs_temp.png"),
    )

    plot_predicted_curves(res_a, df, os.path.join(out_dir, "predicted_curves_additive.png"))
    plot_predicted_curves(res_b, df, os.path.join(out_dir, "predicted_curves_interaction.png"))

    # Added: logistic regression outputs
    df_runs = expand_to_run_level(df)
    df_runs.to_csv(os.path.join(out_dir, "run_level_expanded.csv"), index=False)

    logit_name_a, logit_res_a = fit_logistic_regression(df_runs, interaction=False)
    save_coef_table(
        logit_res_a,
        os.path.join(out_dir, f"{logit_name_a}_coef.csv"),
        os.path.join(out_dir, f"{logit_name_a}_summary.txt"),
    )
    save_logit_test_table(
        logit_res_a,
        os.path.join(out_dir, f"{logit_name_a}_wald_tests.csv"),
    )
    save_predicted_probability_table(
        logit_res_a,
        df_runs,
        os.path.join(out_dir, f"{logit_name_a}_predicted_probs.csv"),
    )
    plot_logit_predictions_with_observed(
        logit_res_a,
        summary_cfg,
        df_runs,
        os.path.join(out_dir, f"{logit_name_a}_predicted_plot.png"),
    )

    logit_name_b, logit_res_b = fit_logistic_regression(df_runs, interaction=True)
    save_coef_table(
        logit_res_b,
        os.path.join(out_dir, f"{logit_name_b}_coef.csv"),
        os.path.join(out_dir, f"{logit_name_b}_summary.txt"),
    )
    save_logit_test_table(
        logit_res_b,
        os.path.join(out_dir, f"{logit_name_b}_wald_tests.csv"),
    )
    save_predicted_probability_table(
        logit_res_b,
        df_runs,
        os.path.join(out_dir, f"{logit_name_b}_predicted_probs.csv"),
    )
    plot_logit_predictions_with_observed(
        logit_res_b,
        summary_cfg,
        df_runs,
        os.path.join(out_dir, f"{logit_name_b}_predicted_plot.png"),
    )

    # Added: treatment-focused plots
    plot_metric_by_treatment_grouped_by_temp(
        summary_cfg,
        metric_col="accuracy_mean",
        ylabel="Mean Accuracy",
        title="Accuracy by Treatment (grouped by temperature)",
        out_png=os.path.join(out_dir, "accuracy_by_treatment_grouped_by_temp.png"),
    )

    plot_metric_by_treatment_grouped_by_temp(
        summary_cfg,
        metric_col="strict_stability_rate",
        ylabel="Strict Stability Rate",
        title="Strict Stability by Treatment (grouped by temperature)",
        out_png=os.path.join(out_dir, "stability_by_treatment_grouped_by_temp.png"),
    )

    plot_metric_by_treatment_grouped_by_temp(
        summary_cfg,
        metric_col="entropy_mean_bits",
        ylabel="Mean Answer Entropy (bits)",
        title="Entropy by Treatment (grouped by temperature)",
        out_png=os.path.join(out_dir, "entropy_by_treatment_grouped_by_temp.png"),
    )

    plot_tradeoff_accuracy_vs_stability(
        summary_cfg,
        out_png=os.path.join(out_dir, "tradeoff_accuracy_vs_stability.png"),
    )

    plot_heatmap_treatment_temp(
        summary_cfg,
        value_col="accuracy_mean",
        title="Heatmap: Accuracy (treatment x temperature)",
        out_png=os.path.join(out_dir, "heatmap_accuracy.png"),
    )
    plot_heatmap_treatment_temp(
        summary_cfg,
        value_col="strict_stability_rate",
        title="Heatmap: Strict Stability (treatment x temperature)",
        out_png=os.path.join(out_dir, "heatmap_stability.png"),
    )
    plot_heatmap_treatment_temp(
        summary_cfg,
        value_col="entropy_mean_bits",
        title="Heatmap: Entropy (treatment x temperature)",
        out_png=os.path.join(out_dir, "heatmap_entropy.png"),
    )

    save_treatment_only_summary(
        summary_cfg,
        out_csv=os.path.join(out_dir, "summary_by_treatment_only.csv"),
    )

    print("Wrote outputs to:", out_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()