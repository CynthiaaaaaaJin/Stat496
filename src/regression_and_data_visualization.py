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
        ax.scatter(sub["entropy_mean_bits"], sub["accuracy_mean"], label=f"temp={temp}")
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
    df["strict_stable"] = df["strict_stable"].astype(str).str.lower().isin(["true", "1", "t", "yes", "y"]).astype(float)

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

    # Plots
    plot_tradeoff(summary_cfg, os.path.join(out_dir, "tradeoff_accuracy_vs_entropy.png"))
    plot_lines_by_temp(summary_cfg, "accuracy_mean", "Accuracy vs Temperature", os.path.join(out_dir, "accuracy_vs_temp.png"))
    plot_lines_by_temp(summary_cfg, "strict_stability_rate", "Strict Stability vs Temperature", os.path.join(out_dir, "stability_vs_temp.png"))
    plot_lines_by_temp(summary_cfg, "entropy_mean_bits", "Entropy vs Temperature", os.path.join(out_dir, "entropy_vs_temp.png"))

    plot_predicted_curves(res_a, df, os.path.join(out_dir, "predicted_curves_additive.png"))
    plot_predicted_curves(res_b, df, os.path.join(out_dir, "predicted_curves_interaction.png"))

    print("Wrote outputs to:", out_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()