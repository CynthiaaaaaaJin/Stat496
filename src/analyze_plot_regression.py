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
    return out.sort_values(["treatment", "temp"]).reset_index(drop=True)


# ============================================================
# GLM (Binomial, logit link) on per-question aggregated outcomes
# ============================================================

def fit_binomial_glm(df_perq: pd.DataFrame, model_type: str):
    """
    Binomial GLM logit with aggregated trials per question:
      successes ~ Binomial(k_runs, p)
      logit(p) = ...
    Implemented as GLM with response = successes/k_runs and freq_weights=k_runs.

    model_type:
      - "additive":  prop ~ C(treatment) + temp
      - "interaction": prop ~ C(treatment) * temp
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
    """
    Expand per-question aggregated results into run-level binary outcomes.

    Example:
      k_runs=5, successes=3 -> rows: [1,1,1,0,0]
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

    # set T0 as reference if present
    if "T0" in set(df_runs["treatment"].astype(str)):
        df_runs["treatment"] = pd.Categorical(df_runs["treatment"])
        cats = list(df_runs["treatment"].cat.categories)
        if "T0" in cats:
            cats = ["T0"] + [c for c in cats if c != "T0"]
            df_runs["treatment"] = df_runs["treatment"].cat.reorder_categories(cats, ordered=False)

    return df_runs


def fit_logistic_regression(df_runs: pd.DataFrame, model_type: str):
    """
    Run-level logistic regression with question-clustered robust SE.

    model_type:
      - "additive":  correct ~ C(treatment) + temp
      - "interaction": correct ~ C(treatment) * temp

    """
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


def save_predicted_probability_table(res, df_source: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    """
    Save predicted probabilities on a grid of (treatment x observed temps).
    """
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
# Tradeoff plots (colored by temperature consistently)
# ============================================================

def _temp_color_map(temps):
    temps = sorted([float(t) for t in temps])
    cmap = plt.get_cmap("tab10")
    return {t: cmap(i % 10) for i, t in enumerate(temps)}


def plot_tradeoff_accuracy_vs_entropy(summary_cfg: pd.DataFrame, out_png: str) -> None:
    ensure_dir(out_png)
    dfp = summary_cfg.copy()
    dfp["temp"] = dfp["temp"].astype(float)

    temps = sorted(dfp["temp"].dropna().unique().tolist())
    color_map = _temp_color_map(temps)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for temp in temps:
        sub = dfp[dfp["temp"] == temp]
        ax.scatter(
            sub["entropy_mean_bits"],
            sub["accuracy_mean"],
            label=f"temp={temp}",
            alpha=0.75,
            color=color_map[temp],
        )
        for _, r in sub.iterrows():
            ax.annotate(
                f"{r['treatment']}_t{r['temp']}",
                (r["entropy_mean_bits"], r["accuracy_mean"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Mean Answer Entropy (bits)")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Tradeoff: Accuracy vs Entropy (colored by temperature)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def plot_tradeoff_accuracy_vs_stability(summary_cfg: pd.DataFrame, out_png: str) -> None:
    ensure_dir(out_png)
    dfp = summary_cfg.copy()
    dfp["temp"] = dfp["temp"].astype(float)

    temps = sorted(dfp["temp"].dropna().unique().tolist())
    color_map = _temp_color_map(temps)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for temp in temps:
        sub = dfp[dfp["temp"] == temp]
        ax.scatter(
            sub["strict_stability_rate"],
            sub["accuracy_mean"],
            label=f"temp={temp}",
            alpha=0.75,
            color=color_map[temp],
        )
        for _, r in sub.iterrows():
            ax.annotate(
                f"{r['treatment']}_t{r['temp']}",
                (r["strict_stability_rate"], r["accuracy_mean"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Strict Stability Rate")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Tradeoff: Accuracy vs Strict Stability (colored by temperature)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


# ============================================================
# Forest plots
# ============================================================

def plot_forest_treatment_or_from_coef_csv(
    coef_csv: str,
    out_png: str,
    title: str,
    include_temp: bool = False,
) -> None:
    """
    Forest plot of OR (95% CI) for treatment indicators relative to T0.
    Reads *_coef.csv produced by save_coef_table.

    include_temp=True will also include the 'temp' row (OR per +1.0 temperature unit).
    """
    ensure_dir(out_png)
    df = pd.read_csv(coef_csv)

    keep = df["term"].astype(str).str.contains(r"C\(treatment\)")
    if include_temp:
        keep = keep | (df["term"].astype(str) == "temp")
    df = df[keep].copy()

    label_map = {
        "C(treatment)[T.T1]": "T1 vs T0",
        "C(treatment)[T.T2]": "T2 vs T0",
        "C(treatment)[T.T3]": "T3 vs T0",
        "C(treatment)[T.T4]": "T4 vs T0",
        "C(treatment)[T.T5]": "T5 vs T0",
        "temp": "Temperature (per +1.0)",
    }
    df["label"] = df["term"].map(label_map).fillna(df["term"].astype(str))

    order = ["T1 vs T0", "T2 vs T0", "T3 vs T0", "T4 vs T0", "T5 vs T0"]
    if include_temp:
        order = order + ["Temperature (per +1.0)"]
    df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)
    df = df.sort_values("label").reset_index(drop=True)

    if len(df) == 0:
        return

    y = np.arange(len(df))
    or_ = df["odds_ratio"].to_numpy(dtype=float)
    low = df["or_ci_low"].to_numpy(dtype=float)
    high = df["or_ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([or_ - low, high - or_])

    fig = plt.figure(figsize=(7, 4.6))
    ax = fig.add_subplot(111)
    ax.errorbar(or_, y, xerr=xerr, fmt="o", capsize=4)
    ax.axvline(1.0, linestyle="--")

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"].astype(str).tolist())
    ax.set_xlabel("Odds Ratio (OR) with 95% CI")
    ax.set_title(title)

    if "p_value" in df.columns:
        for i, p in enumerate(df["p_value"].to_numpy(dtype=float)):
            ax.text(high[i] * 1.02, i, f"p={p:.3g}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_forest_temp_slopes_from_interaction_coef_csv(
    interaction_coef_csv: str,
    out_png: str,
    title: str = "Temperature sensitivity by treatment (interaction model)",
) -> None:
    """
    Forest plot for temperature slope ORs per +1.0 temp, by treatment,
    using the interaction model coef csv: * ~ C(treatment)*temp.

    For T0: slope = temp
    For Tj: slope = temp + C(treatment)[T.Tj]:temp

    CI is approximate (ignores covariance), good for visualization.
    """
    ensure_dir(out_png)
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
        rows.append({"label": f"{t}: OR per +1.0 temp", "odds_ratio": or_, "low": low, "high": high})

    out = pd.DataFrame(rows)
    if out.empty:
        return

    y = np.arange(len(out))
    or_ = out["odds_ratio"].to_numpy()
    low = out["low"].to_numpy()
    high = out["high"].to_numpy()
    xerr = np.vstack([or_ - low, high - or_])

    fig = plt.figure(figsize=(7, 4.6))
    ax = fig.add_subplot(111)
    ax.errorbar(or_, y, xerr=xerr, fmt="o", capsize=4)
    ax.axvline(1.0, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(out["label"].astype(str).tolist())
    ax.set_xlabel("Odds Ratio (OR) per +1.0 temperature (approx. 95% CI)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
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

    # Config-level summary for tradeoff plots
    summary_cfg = summarize_config(df)
    summary_path = os.path.join(out_dir, "summary_by_config.csv")
    summary_cfg.to_csv(summary_path, index=False)

    # =========================
    # GLM models (KEEP)
    # =========================
    for model_type in ["additive", "interaction"]:
        name, res = fit_binomial_glm(df, model_type=model_type)
        save_coef_table(
            res,
            os.path.join(out_dir, f"{name}_coef.csv"),
            os.path.join(out_dir, f"{name}_summary.txt"),
        )

    # =========================
    # Run-level logistic models
    # =========================
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

    # =========================
    # Tradeoff plots 
    # =========================
    plot_tradeoff_accuracy_vs_entropy(
        summary_cfg,
        os.path.join(out_dir, "tradeoff_accuracy_vs_entropy.png"),
    )
    plot_tradeoff_accuracy_vs_stability(
        summary_cfg,
        os.path.join(out_dir, "tradeoff_accuracy_vs_stability.png"),
    )

    # =========================
    # Forest plots (KEEP; from run-level logit outputs)
    # =========================
    plot_forest_treatment_or_from_coef_csv(
        coef_csv=os.path.join(out_dir, "logit_additive_coef.csv"),
        out_png=os.path.join(out_dir, "forest_logit_additive_treatment_or.png"),
        title="Treatment effects on correctness (logit additive; clustered SE)",
        include_temp=False,
    )

    plot_forest_treatment_or_from_coef_csv(
        coef_csv=os.path.join(out_dir, "logit_interaction_coef.csv"),
        out_png=os.path.join(out_dir, "forest_logit_interaction_treatment_or.png"),
        title="Treatment effects on correctness (logit interaction; clustered SE)",
        include_temp=False,
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