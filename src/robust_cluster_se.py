#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def split_config_id(config_id: str) -> Tuple[str, float]:
    # expects: T0_temp0.2
    if isinstance(config_id, str) and "_temp" in config_id:
        t, temp = config_id.split("_temp", 1)
        try:
            return t, float(temp)
        except ValueError:
            return t, float("nan")
    return str(config_id), float("nan")


def ensure_dir_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def save_outputs(res, out_prefix: str) -> None:
    """
    Save summary txt + coef csv (OR + CI) using res.bse / res.pvalues (already cluster-robust).
    """
    out_txt = out_prefix + "_summary.txt"
    out_csv = out_prefix + "_coef.csv"
    ensure_dir_for_file(out_txt)
    ensure_dir_for_file(out_csv)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(res.summary().as_text())

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
            "std_err_cluster": se.values,
            "p_value_cluster": p.values,
            "odds_ratio": or_.values,
            "or_ci_low": ci_low.values,
            "or_ci_high": ci_high.values,
        }
    )
    tab.to_csv(out_csv, index=False)


def fit_glm_cluster(df: pd.DataFrame, formula: str, groups: pd.Series):
    """
    Fit binomial GLM with freq_weights and CLUSTER-ROBUST SE by question_id.
    We fit directly with cov_type='cluster' because some statsmodels versions
    return None from _get_robustcov_results().
    """
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["k_runs"].astype(float),
    )

    # Most statsmodels versions support cov_type/cov_kwds in fit()
    try:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": groups.astype(str)})
    except TypeError:
        # Fallback: fit normally (non-robust) with a clear message
        # (Should be rare in your env because you already used cov_type in run-level fit.)
        res = model.fit()
        raise RuntimeError(
            "Your statsmodels version does not support cov_type='cluster' in GLM.fit(). "
            "Please upgrade statsmodels, or use a different robust covariance method."
        )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-question-csv", required=True)
    ap.add_argument("--out-dir", default="outputs/robust_cluster")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.per_question_csv)

    # Parse treatment/temp from config_id
    tt = df["config_id"].apply(lambda x: split_config_id(str(x)))
    df["treatment"] = tt.apply(lambda x: x[0])
    df["temp"] = tt.apply(lambda x: x[1])

    # Numeric cleanup
    for c in ["k_runs", "accuracy_over_runs", "temp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    required = ["config_id", "question_id", "treatment", "temp", "k_runs", "accuracy_over_runs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in per-question csv: {missing}")

    df = df.dropna(subset=["question_id", "temp", "k_runs", "accuracy_over_runs"]).copy()
    df["k_runs"] = df["k_runs"].astype(int)

    # successes = round(accuracy * k)
    df["successes"] = (df["accuracy_over_runs"] * df["k_runs"]).round().astype(int)
    df["successes"] = df["successes"].clip(lower=0, upper=df["k_runs"])

    # Binomial proportion response
    df["prop"] = df["successes"] / df["k_runs"]

    # Set baseline treatment T0 if present
    if "T0" in set(df["treatment"].astype(str)):
        df["treatment"] = pd.Categorical(df["treatment"])
        cats = list(df["treatment"].cat.categories)
        if "T0" in cats:
            cats = ["T0"] + [c for c in cats if c != "T0"]
            df["treatment"] = df["treatment"].cat.reorder_categories(cats, ordered=False)

    groups = df["question_id"].astype(str)

    # -------- Additive (cluster-robust) --------
    res_add = fit_glm_cluster(df, "prop ~ C(treatment) + temp", groups)
    save_outputs(res_add, os.path.join(args.out_dir, "glm_additive_cluster"))

    # -------- Interaction (cluster-robust) --------
    res_int = fit_glm_cluster(df, "prop ~ C(treatment) * temp", groups)
    save_outputs(res_int, os.path.join(args.out_dir, "glm_interaction_cluster"))

    print("Wrote cluster-robust outputs to:", args.out_dir)
    print(" - glm_additive_cluster_summary.txt / glm_additive_cluster_coef.csv")
    print(" - glm_interaction_cluster_summary.txt / glm_interaction_cluster_coef.csv")


if __name__ == "__main__":
    main()