import argparse
import json
import re
from collections import defaultdict

import pandas as pd


FINAL_RE = re.compile(r"Final:\s*([A-E])", re.IGNORECASE)


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_answers_jsonl(path: str):
    ans = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ans[obj["qid"]] = obj["answer"].upper()
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="outputs/rq1_small_outputs.jsonl")
    ap.add_argument("--answers", default="data/apchem_answers_sample_10.jsonl")
    args = ap.parse_args()

    rows = read_jsonl(args.preds)
    ans = read_answers_jsonl(args.answers)

    df = pd.DataFrame(rows)
    df["gold"] = df["qid"].map(ans)
    df["correct"] = (df["pred"] == df["gold"]).astype(int)

    # token cost proxy (Gemini provides total_token_count, GPT4All may be missing)
    def get_total_tokens(u):
        if not isinstance(u, dict):
            return None
        return u.get("total_token_count")

    df["total_tokens"] = df["usage"].apply(get_total_tokens)

    # Accuracy per condition
    acc = df.groupby(["treatment", "temperature"])["correct"].mean().reset_index()
    acc = acc.sort_values(["temperature", "treatment"])

    # Stability: per qid/treatment/temp, check if all repeats same pred
    stab_rows = []
    for (qid, tr, temp), sub in df.groupby(["qid", "treatment", "temperature"]):
        preds = list(sub["pred"])
        stable = int(len(set(preds)) == 1 and preds[0] != "")
        stab_rows.append({"qid": qid, "treatment": tr, "temperature": temp, "stable": stable})

    stab = pd.DataFrame(stab_rows).groupby(["treatment", "temperature"])["stable"].mean().reset_index()

    # Avg token cost (if available)
    tok = df.groupby(["treatment", "temperature"])["total_tokens"].mean().reset_index()

    merged = acc.merge(stab, on=["treatment", "temperature"], how="left").merge(tok, on=["treatment", "temperature"], how="left")
    merged = merged.rename(columns={"correct": "accuracy", "stable": "stability", "total_tokens": "avg_total_tokens"})

    print("\n=== RQ1 Small Test Summary ===")
    print(merged.to_string(index=False))

    # Variety check: distribution of predicted letters
    print("\n=== Variety Check: Pred letter distribution ===")
    dist = df.groupby(["treatment", "temperature", "pred"]).size().reset_index(name="count")
    print(dist.to_string(index=False))


if __name__ == "__main__":
    main()
