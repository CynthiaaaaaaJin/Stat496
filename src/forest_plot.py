import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/krise/Documents/GitHub/Stat496/outputs/blog100_glm_new_k5_all_temps/logit_additive_coef.csv")

# keep only treatment terms (drop intercept and temp)
df = df[df["term"].str.contains(r"C\(treatment\)")].copy()

# nicer labels
label_map = {
    "C(treatment)[T.T1]": "T1 vs T0",
    "C(treatment)[T.T2]": "T2 vs T0",
    "C(treatment)[T.T3]": "T3 vs T0",
    "C(treatment)[T.T4]": "T4 vs T0",
    "C(treatment)[T.T5]": "T5 vs T0",
}
df["label"] = df["term"].map(label_map)

# order top->bottom (T5 last or first; adjust as you like)
order = ["T1 vs T0", "T2 vs T0", "T3 vs T0", "T4 vs T0", "T5 vs T0"]
df["label"] = pd.Categorical(df["label"], categories=order, ordered=True)
df = df.sort_values("label")

# === 2) plotting positions ===
y = range(len(df))

or_ = df["odds_ratio"].to_numpy()
low = df["or_ci_low"].to_numpy()
high = df["or_ci_high"].to_numpy()

# xerr expects distance to low/high
xerr = [or_ - low, high - or_]

# === 3) plot ===
plt.figure(figsize=(7, 4.5))
plt.errorbar(or_, y, xerr=xerr, fmt='o', capsize=4)

# reference line at OR=1
plt.axvline(1.0, linestyle='--')

plt.yticks(list(y), df["label"])
plt.xlabel("Odds Ratio (OR) with 95% CI")
plt.title("Treatment effects on correctness (logistic regression)")


for i, p in enumerate(df["p_value"].to_numpy()):
    plt.text(high[i] * 1.02, i, f"p={p:.3g}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("forest_treatment_or.png", dpi=300)
plt.show()