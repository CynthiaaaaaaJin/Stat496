from __future__ import annotations
import argparse
import json
from typing import List, Dict, Any

import pandas as pd

def read_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", default="outputs/runs.jsonl")
    ap.add_argument("--in-summary-csv", default="outputs/summary.csv")
    ap.add_argument("--in-per-question-csv", default="outputs/per_question.csv")
    ap.add_argument("--out-md", default="outputs/writeup.md")
    ap.add_argument("--title", default="Prompt × Temperature Experiment")
    args = ap.parse_args()

    rows = read_jsonl_rows(args.in_jsonl)
    summary = pd.read_csv(args.in_summary_csv)
    per_q = pd.read_csv(args.in_per_question_csv)

    df = pd.DataFrame(rows).sort_values(["config_id", "question_id", "run_id"])
    sample = df.groupby(["config_id", "question_id"]).head(1)

    lines: List[str] = []
    lines.append(f"# {args.title}\n")
    lines.append("## Goal\n")
    lines.append("Measure how different instruction styles (prompt treatments) and temperatures affect **accuracy**, **stability**, and (when available) **token cost**.\n")

    lines.append("## Treatments\n")
    lines.append("- T0: Normal answering\n")
    lines.append("- T1: Only final result\n")
    lines.append("- T2: Step-by-step reasoning\n")
    lines.append("- T3: Restrict to provided context (MCQ options) / avoid outside assumptions\n")
    lines.append("- T4: Step-by-step + restricted context\n")
    lines.append("- T5: Self-check then commit (self-check hidden)\n")

    lines.append("## Metrics\n")
    lines.append("- Accuracy: correct / total\n")
    lines.append("- Strict stability: for each question, all K runs give the same final answer\n")
    lines.append("- Mode frequency: the fraction of runs that match the most common answer (continuous stability)\n")
    lines.append("- Entropy: answer distribution entropy (higher = more random)\n")
    lines.append("- Token cost (if logged): total_tokens, tokens_per_correct\n")

    lines.append("## Results Summary (per config)\n")
    lines.append(summary.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Per-question Snapshot (first 30 rows)\n")
    lines.append(per_q.head(30).to_markdown(index=False))
    lines.append("\n")

    lines.append("## Example Outputs (one sample per config × question)\n")
    for _, r in sample.iterrows():
        lines.append(f"### {r['config_id']} — {r['question_id']}\n")
        lines.append(f"- Parsed: `{r['parsed_answer']}` | Correct: `{r['correct']}`\n")
        p = str(r.get("prompt",""))
        if len(p) > 700: p = p[:700] + " ...[truncated]"
        o = str(r.get("raw_output",""))
        if len(o) > 500: o = o[:500] + " ...[truncated]"
        lines.append("Prompt (truncated):\n```text\n" + p + "\n```\n")
        lines.append("Output (truncated):\n```text\n" + o + "\n```\n")

    import os
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.out_md}")

if __name__ == "__main__":
    main()
