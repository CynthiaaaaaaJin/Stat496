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
    Reduce per-question rows to config-level means (for tradeoff plots).
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

    out["treatment"] = out["treatment"].astype(str)
    out["temp"] = out["temp"].astype(float)
    return out.sort_values(["treatment", "temp"]).reset_index(drop=True)


def summarize_metric_for_barplot(
    df_perq: pd.DataFrame,
    metric_col: str,
    out_mean_name: str = "mean_value",
) -> pd.DataFrame:
    """
    Compute grouped means and error bars across questions for each treatment x temp.

    Error bars:
    - se   = standard error
    - ci95 = 1.96 * se
    """
    tmp = df_perq.copy()
    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")

    g = tmp.groupby(["treatment", "temp"], dropna=False)[metric_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": out_mean_name, "std": "sd", "count": "n"})
    out["se"] = out["sd"] / np.sqrt(out["n"])
    out["ci95"] = 1.96 * out["se"]

    out["treatment"] = out["treatment"].astype(str)
    out["temp"] = out["temp"].astype(float)
    return out.sort_values(["treatment", "temp"]).reset_index(drop=True)


# ============================================================
# GLM (Binomial, logit link) on per-question aggregated outcomes
# ============================================================

def fit_binomial_glm(df_perq: pd.DataFrame, model_type: str):
    """
    Binomial GLM logit with aggregated trials per question:
        successes ~ Binomial(k_runs, p)
        logit(p) = ...

    Implemented as GLM with response = successes/k_runs and freq_weights = k_runs.

    model_type:
      - "additive":    prop ~ C(treatment) + temp
      - "interaction": prop ~ C(treatment) * temp
    """
    df = df_perq.copy()
    df["prop"] = df["successes"] / df["k_runs"]

    if "T0" in set(df["treatment"].astype(str)):
        df["treatment"] = pd.Categorical(df["treatment"])
        cats = list(df["treatment"].cat.categories)
        if "T0" in cats:
            cats = ["T0"] + [c for c in cats if c != "T0"]
        df["treatment"] = df["treatment"].cat.reorder_categories(cats, ordered=False)

    if model_type == "additive":
        formula = "prop ~ C(treatment) + temp"
        name = "glm_additive"
    elif model_type == "interaction":
        formula = "prop ~ C(treatment) * temp"
        name = "glm_interaction"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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


# ============================================================
# Run-level Logistic regression (GLM Binomial) with clustered SE
# ============================================================

def expand_to_run_level(df_perq: pd.DataFrame) -> pd.DataFrame:
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


def fit_logistic_regression(df_runs: pd.DataFrame, model_type: str):
    if model_type == "additive":
        formula = "correct ~ C(treatment) + temp"
        name = "logit_additive"
    elif model_type == "interaction":
        formula = "correct ~ C(treatment) * temp"
        name = "logit_interaction"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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


def save_predicted_probability_table(
    res,
    df_source: pd.DataFrame,
    out_csv: str,
) -> pd.DataFrame:
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


# ============================================================
# Tradeoff plots
# ============================================================

def _temp_color_map(temps):
    temps = sorted([float(t) for t in temps])
    cmap = plt.get_cmap("viridis")
    return {t: cmap(i / max(len(temps) - 1, 1)) for i, t in enumerate(temps)}


def _pareto_mask_max2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i]))
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


def _pareto_mask_entropy(x_entropy: np.ndarray, y_acc: np.ndarray) -> np.ndarray:
    n = len(x_entropy)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (
            (x_entropy <= x_entropy[i])
            & (y_acc >= y_acc[i])
            & ((x_entropy < x_entropy[i]) | (y_acc > y_acc[i]))
        )
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


def plot_tradeoff_accuracy_vs_entropy(summary_cfg: pd.DataFrame, out_png: str) -> None:
    ensure_dir(out_png)

    dfp = summary_cfg.copy()
    dfp["temp"] = dfp["temp"].astype(float)
    dfp["entropy_mean_bits"] = dfp["entropy_mean_bits"].astype(float)
    dfp["accuracy_mean"] = dfp["accuracy_mean"].astype(float)

    temps = sorted(dfp["temp"].dropna().unique().tolist())
    color_map = _temp_color_map(temps)

    x = dfp["entropy_mean_bits"].to_numpy()
    y = dfp["accuracy_mean"].to_numpy()
    best_mask = _pareto_mask_entropy(x_entropy=x, y_acc=y)

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111)

    for temp in temps:
        sub = dfp[(dfp["temp"] == temp) & (~best_mask)]
        ax.scatter(
            sub["entropy_mean_bits"],
            sub["accuracy_mean"],
            label=f"temp={temp}",
            alpha=0.40,
            color=color_map[temp],
            s=80,
            edgecolors="white",
            linewidths=0.8,
        )

    best = dfp[best_mask]
    for temp in temps:
        sub = best[best["temp"] == temp]
        ax.scatter(
            sub["entropy_mean_bits"],
            sub["accuracy_mean"],
            alpha=0.95,
            color=color_map[temp],
            s=120,
            edgecolors="black",
            linewidths=0.8,
        )

    for _, r in dfp.iterrows():
        ax.annotate(
            f"{r['treatment']}_t{r['temp']}",
            (r["entropy_mean_bits"], r["accuracy_mean"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Mean Answer Entropy (bits)")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Tradeoff: Accuracy vs Entropy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", linestyle="--", alpha=0.20)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_accuracy_vs_stability(summary_cfg: pd.DataFrame, out_png: str) -> None:
    ensure_dir(out_png)

    dfp = summary_cfg.copy()
    dfp["temp"] = dfp["temp"].astype(float)
    dfp["strict_stability_rate"] = dfp["strict_stability_rate"].astype(float)
    dfp["accuracy_mean"] = dfp["accuracy_mean"].astype(float)

    temps = sorted(dfp["temp"].dropna().unique().tolist())
    color_map = _temp_color_map(temps)

    x = dfp["strict_stability_rate"].to_numpy()
    y = dfp["accuracy_mean"].to_numpy()
    best_mask = _pareto_mask_max2(x=x, y=y)

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111)

    for temp in temps:
        sub = dfp[(dfp["temp"] == temp) & (~best_mask)]
        ax.scatter(
            sub["strict_stability_rate"],
            sub["accuracy_mean"],
            label=f"temp={temp}",
            alpha=0.40,
            color=color_map[temp],
            s=80,
            edgecolors="white",
            linewidths=0.8,
        )

    best = dfp[best_mask]
    for temp in temps:
        sub = best[best["temp"] == temp]
        ax.scatter(
            sub["strict_stability_rate"],
            sub["accuracy_mean"],
            alpha=0.95,
            color=color_map[temp],
            s=120,
            edgecolors="black",
            linewidths=0.8,
        )

    for _, r in dfp.iterrows():
        ax.annotate(
            f"{r['treatment']}_t{r['temp']}",
            (r["strict_stability_rate"], r["accuracy_mean"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Strict Stability Rate")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Tradeoff: Accuracy vs Strict Stability")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", linestyle="--", alpha=0.20)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# BAR CHARTS WITH ERROR BARS
# ============================================================

def _ordered_treatments(summary_df: pd.DataFrame):
    canonical = ["T0", "T1", "T2", "T3", "T4", "T5"]
    present = [t for t in canonical if t in set(summary_df["treatment"].astype(str))]
    return present if present else sorted(summary_df["treatment"].astype(str).unique().tolist())


def _ordered_temps(summary_df: pd.DataFrame):
    temps = summary_df["temp"].dropna().astype(float).unique().tolist()
    return sorted(temps)


def plot_metric_by_treatment_grouped_by_temp_bar(
    bar_df: pd.DataFrame,
    mean_col: str,
    err_col: str,
    ylabel: str,
    title: str,
    out_png: str,
) -> None:
    """
    Grouped bar chart with error bars.

    X-axis: treatment
    Groups: temperature
    Error bars: err_col (e.g., se or ci95)

    Uses a colorful viridis-like palette similar to the screenshot style.
    """
    ensure_dir(out_png)

    dfp = bar_df.copy()
    dfp["treatment"] = dfp["treatment"].astype(str)
    dfp["temp"] = dfp["temp"].astype(float)

    treatments = _ordered_treatments(dfp)
    temps = _ordered_temps(dfp)

    x = np.arange(len(treatments))
    width = 0.78 / max(len(temps), 1)

    cmap = plt.get_cmap("viridis")
    if len(temps) == 1:
        pretty_colors = [cmap(0.55)]
    else:
        pretty_colors = [cmap(v) for v in np.linspace(0.08, 0.95, len(temps))]

    fig = plt.figure(figsize=(8.8, 5.2))
    ax = fig.add_subplot(111)

    for i, temp in enumerate(temps):
        sub = dfp[dfp["temp"] == temp]
        heights = []
        yerr = []

        for t in treatments:
            row = sub[sub["treatment"] == t]
            if len(row):
                heights.append(float(row[mean_col].iloc[0]))
                yerr.append(float(row[err_col].iloc[0]))
            else:
                heights.append(np.nan)
                yerr.append(np.nan)

        xpos = x - 0.39 + width / 2 + i * width

        ax.bar(
            xpos,
            heights,
            width=width,
            yerr=yerr,
            capsize=4,
            label=f"temp={temp}",
            color=pretty_colors[i],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.98,
            error_kw={
                "elinewidth": 1.1,
                "ecolor": "#3a3a3a",
                "capsize": 4,
                "capthick": 1.1,
            },
        )

    ax.set_xticks(x)
    ax.set_xticklabels(treatments, fontsize=11)
    ax.set_xlabel("Treatment", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.22)

    ax.tick_params(axis="y", labelsize=10)
    ax.legend(title="Temperature", frameon=False, fontsize=10, title_fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Forest plots
# ============================================================

def plot_forest_treatment_or_from_coef_csv(
    coef_csv: str,
    out_png: str,
    title: str,
) -> None:
    ensure_dir(out_png)

    if not os.path.exists(coef_csv):
        return

    df = pd.read_csv(coef_csv)
    df = df[df["term"].astype(str).str.contains(r"C\(treatment\)")].copy()

    label_map = {
        "C(treatment)[T.T1]": "T1 vs T0",
        "C(treatment)[T.T2]": "T2 vs T0",
        "C(treatment)[T.T3]": "T3 vs T0",
        "C(treatment)[T.T4]": "T4 vs T0",
        "C(treatment)[T.T5]": "T5 vs T0",
    }

    df["label"] = df["term"].map(label_map).fillna(df["term"].astype(str))
    order = ["T1 vs T0", "T2 vs T0", "T3 vs T0", "T4 vs T0", "T5 vs T0"]
    df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)
    df = df.sort_values("label").reset_index(drop=True)

    if len(df) == 0:
        return

    y = np.arange(len(df))
    or_ = df["odds_ratio"].to_numpy(dtype=float)
    low = df["or_ci_low"].to_numpy(dtype=float)
    high = df["or_ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([or_ - low, high - or_])

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)

    ax.errorbar(or_, y, xerr=xerr, fmt="o", capsize=4, color="#2f5d8a", ecolor="#5b7ea4")
    ax.axvline(1.0, linestyle="--", color="#666666")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"].astype(str).tolist())
    ax.set_xlabel("Odds Ratio (OR) with 95% CI")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.20)

    for i, p in enumerate(df["p_value"].to_numpy(dtype=float)):
        ax.text(high[i] * 1.02, i, f"p={p:.3g}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_forest_temp_slopes_from_interaction_coef_csv(
    interaction_coef_csv: str,
    out_png: str,
    title: str = "Temperature sensitivity by treatment (interaction logit; approx. CI)",
) -> None:
    ensure_dir(out_png)

    if not os.path.exists(interaction_coef_csv):
        return

    df = pd.read_csv(interaction_coef_csv)
    coef = dict(zip(df["term"].astype(str), df["coef_logit"].astype(float)))
    se = dict(zip(df["term"].astype(str), df["std_err"].astype(float)))

    treatments = ["T0", "T1", "T2", "T3", "T4", "T5"]
    rows = []

    for t in treatments:
        if t == "T0":
            b = coef.get("temp", np.nan)
            s = se.get("temp", np.nan)
        else:
            inter = f"C(treatment)[T.{t}]:temp"
            b = coef.get("temp", np.nan) + coef.get(inter, 0.0)
            s = np.sqrt((se.get("temp", np.nan) ** 2) + (se.get(inter, 0.0) ** 2))

        if not np.isfinite(b) or not np.isfinite(s):
            continue

        or_ = np.exp(b)
        low = np.exp(b - 1.96 * s)
        high = np.exp(b + 1.96 * s)
        rows.append({"label": f"{t}: OR per +1.0 temp", "or": or_, "low": low, "high": high})

    out = pd.DataFrame(rows)
    if out.empty:
        return

    y = np.arange(len(out))
    or_ = out["or"].to_numpy()
    low = out["low"].to_numpy()
    high = out["high"].to_numpy()
    xerr = np.vstack([or_ - low, high - or_])

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)

    ax.errorbar(or_, y, xerr=xerr, fmt="o", capsize=4, color="#2f5d8a", ecolor="#5b7ea4")
    ax.axvline(1.0, linestyle="--", color="#666666")
    ax.set_yticks(list(y))
    ax.set_yticklabels(out["label"].astype(str).tolist())
    ax.set_xlabel("Odds Ratio (OR) per +1.0 temperature (approx. 95% CI)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.20)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-question-csv", required=True, help="per-question CSV (question x config)")
    ap.add_argument("--out-dir", default="outputs/glm_analysis_pruned")
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

    # Build binomial counts per question x config
    df["successes"] = (df["accuracy_over_runs"] * df["k_runs"]).round().astype(int)
    df["successes"] = df["successes"].clip(lower=0, upper=df["k_runs"].astype(int))
    df = df.dropna(subset=["temp", "k_runs", "successes"])

    # Config-level summary
    summary_cfg = summarize_config(df)
    summary_path = os.path.join(out_dir, "summary_by_config.csv")
    summary_cfg.to_csv(summary_path, index=False)

    # Group summaries for bar charts with error bars
    acc_bar = summarize_metric_for_barplot(
        df,
        metric_col="accuracy_over_runs",
        out_mean_name="accuracy_mean",
    )
    ent_bar = summarize_metric_for_barplot(
        df,
        metric_col="answer_entropy_bits",
        out_mean_name="entropy_mean_bits",
    )
    stab_bar = summarize_metric_for_barplot(
        df,
        metric_col="strict_stable",
        out_mean_name="strict_stability_rate",
    )

    # GLM models
    for model_type in ["additive", "interaction"]:
        name, res = fit_binomial_glm(df, model_type=model_type)
        save_coef_table(
            res,
            os.path.join(out_dir, f"{name}_coef.csv"),
            os.path.join(out_dir, f"{name}_summary.txt"),
        )

    # Run-level logistic models
    df_runs = expand_to_run_level(df)
    df_runs.to_csv(os.path.join(out_dir, "run_level_expanded.csv"), index=False)

    for model_type in ["additive", "interaction"]:
        name, res = fit_logistic_regression(df_runs, model_type=model_type)
        save_coef_table(
            res,
            os.path.join(out_dir, f"{name}_coef.csv"),
            os.path.join(out_dir, f"{name}_summary.txt"),
        )
        save_logit_test_table(
            res,
            os.path.join(out_dir, f"{name}_wald_tests.csv"),
        )
        save_predicted_probability_table(
            res,
            df_runs,
            os.path.join(out_dir, f"{name}_predicted_probs.csv"),
        )

    # Tradeoff plots
    plot_tradeoff_accuracy_vs_entropy(
        summary_cfg,
        os.path.join(out_dir, "tradeoff_accuracy_vs_entropy.png"),
    )
    plot_tradeoff_accuracy_vs_stability(
        summary_cfg,
        os.path.join(out_dir, "tradeoff_accuracy_vs_stability.png"),
    )

    # Bar charts with colorful palette
    # Change err_col from "se" to "ci95" if you want 95% CI instead.
    plot_metric_by_treatment_grouped_by_temp_bar(
        acc_bar,
        mean_col="accuracy_mean",
        err_col="se",
        ylabel="Mean Accuracy",
        title="Accuracy by Treatment (grouped by temperature)",
        out_png=os.path.join(out_dir, "accuracy_by_treatment_grouped_by_temp_BAR.png"),
    )
    plot_metric_by_treatment_grouped_by_temp_bar(
        ent_bar,
        mean_col="entropy_mean_bits",
        err_col="se",
        ylabel="Mean Answer Entropy (bits)",
        title="Entropy by Treatment (grouped by temperature)",
        out_png=os.path.join(out_dir, "entropy_by_treatment_grouped_by_temp_BAR.png"),
    )
    plot_metric_by_treatment_grouped_by_temp_bar(
        stab_bar,
        mean_col="strict_stability_rate",
        err_col="se",
        ylabel="Strict Stability Rate",
        title="Strict Stability by Treatment (grouped by temperature)",
        out_png=os.path.join(out_dir, "stability_by_treatment_grouped_by_temp_BAR.png"),
    )

    # Forest plots
    plot_forest_treatment_or_from_coef_csv(
        coef_csv=os.path.join(out_dir, "logit_additive_coef.csv"),
        out_png=os.path.join(out_dir, "forest_logit_additive_treatment_or.png"),
        title="Treatment effects on correctness (logit additive; clustered SE)",
    )
    plot_forest_treatment_or_from_coef_csv(
        coef_csv=os.path.join(out_dir, "logit_interaction_coef.csv"),
        out_png=os.path.join(out_dir, "forest_logit_interaction_treatment_or.png"),
        title="Treatment effects on correctness (logit interaction; clustered SE)",
    )
    plot_forest_temp_slopes_from_interaction_coef_csv(
        interaction_coef_csv=os.path.join(out_dir, "logit_interaction_coef.csv"),
        out_png=os.path.join(out_dir, "forest_logit_interaction_temp_slopes.png"),
        title="Temperature sensitivity by treatment (interaction logit; approx. CI)",
    )

    print("Wrote outputs to:", out_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()