import json
import re
from pathlib import Path

import pandas as pd


FINAL_RE = re.compile(r"Final\s*:\s*([A-E])", re.IGNORECASE)

def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def extract_pred(row) -> str | None:
    # Prefer explicit pred field if present and valid
    pred = row.get("pred", None)
    if isinstance(pred, str):
        pred = pred.strip().upper()
        if pred in ["A","B","C","D","E"]:
            return pred

    # Otherwise parse from raw_text
    raw = row.get("raw_text", "") or ""
    m = FINAL_RE.search(raw)
    if m:
        return m.group(1).upper()

    return None

def main(
    data_path="data/apchem_60.jsonl",
    outputs_path="outputs/rq1_outputs.jsonl",
    out_csv="outputs/analysis_summary.csv",
):
    data_path = Path(data_path)
    outputs_path = Path(outputs_path)

    df_data = read_jsonl(data_path)
    df_out = read_jsonl(outputs_path)

    # ensure types
    df_data["qid"] = df_data["qid"].astype(int)
    df_out["qid"] = df_out["qid"].astype(int)

    # get gold
    if "gold" not in df_data.columns:
        raise ValueError("Your data jsonl must contain a 'gold' field per row (A-E).")

    df_data["gold"] = df_data["gold"].astype(str).str.strip().str.upper()
    df_data.loc[~df_data["gold"].isin(["A","B","C","D","E"]), "gold"] = None

    # extract prediction
    df_out["pred_extracted"] = df_out.apply(lambda r: extract_pred(r), axis=1)

    # merge on qid
    df = df_out.merge(df_data[["qid", "gold"]], on="qid", how="left")

    # compute correctness
    df["is_correct"] = (df["pred_extracted"] == df["gold"])
    df["has_gold"] = df["gold"].notna()
    df["has_pred"] = df["pred_extracted"].notna()

    # -------- summary prints --------
    n = len(df)
    print(f"Rows in outputs: {n}")
    print(f"Rows with gold available: {df['has_gold'].sum()} / {n}")
    print(f"Rows with a parsable prediction: {df['has_pred'].sum()} / {n}")

    # Overall accuracy (only where gold exists and prediction exists)
    valid = df[df["has_gold"] & df["has_pred"]]
    if len(valid) == 0:
        print("No valid rows to score (missing gold or pred).")
        return

    overall_acc = valid["is_correct"].mean()
    print(f"\nOverall accuracy (scorable rows only): {overall_acc:.3f}  (n={len(valid)})")

    # Accuracy by treatment/temp
    group_cols = []
    if "treatment" in df.columns:
        group_cols.append("treatment")
    if "temperature" in df.columns:
        group_cols.append("temperature")

    if group_cols:
        summary = (
            valid.groupby(group_cols)
            .agg(
                n=("is_correct", "size"),
                acc=("is_correct", "mean"),
                missing_pred=("has_pred", lambda s: (~s).sum()),
            )
            .reset_index()
            .sort_values(group_cols)
        )
        print("\nAccuracy by " + ", ".join(group_cols) + ":")
        print(summary.to_string(index=False))
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_csv, index=False)
        print(f"\nSaved summary CSV to: {out_csv}")

    # Optional: show most common failure reasons
    # rows where pred missing or gold missing
    miss_pred = df[df["has_gold"] & (~df["has_pred"])]
    if len(miss_pred) > 0:
        print(f"\nWARNING: {len(miss_pred)} rows had gold but no parsable 'Final: X'.")
        # show a few examples
        cols = [c for c in ["qid","treatment","temperature","raw_text"] if c in df.columns]
        print(miss_pred[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
