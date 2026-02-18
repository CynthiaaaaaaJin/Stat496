from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, Any, List, Optional

IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

def normalize_stem(context: str, question: str) -> str:
    context = (context or "").strip()
    question = (question or "").strip()
    if context and question:
        return f"{context}\n\nQuestion: {question}"
    return question or context

def get_label_index(row: Dict[str, str]) -> Optional[int]:
    # supports either label_index or label
    for key in ("label_index", "label"):
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            try:
                return int(str(row[key]).strip())
            except ValueError:
                return None
    return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True, help="Input CSV path (must have answer0..answer3 and label_index/label).")
    ap.add_argument("--out-jsonl", required=True, help="Output JSONL dataset path.")
    ap.add_argument("--n", type=int, default=0, help="If >0, only write first N rows (e.g., 10).")
    ap.add_argument("--start", type=int, default=0, help="Start row offset (0-based).")
    ap.add_argument("--id-prefix", default="BLOG", help="If id missing/duplicate, use this prefix to build ids.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    items: List[Dict[str, Any]] = []
    with open(args.in_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # slice
    rows = rows[args.start:]
    if args.n and args.n > 0:
        rows = rows[:args.n]

    # first pass: collect ids, but we will force uniqueness
    raw_ids = []
    for i, r in enumerate(rows):
        rid = (r.get("id") or "").strip()
        if not rid:
            rid = f"{args.id_prefix}_{args.start + i:04d}"
        raw_ids.append(rid)

    counts = Counter(raw_ids)

    for i, r in enumerate(rows):
        rid = raw_ids[i]
        # force unique ids: if duplicates exist, append row index
        if counts[rid] > 1:
            rid = f"{rid}_{args.start + i:04d}"

        context = r.get("context", "") or ""
        question = r.get("question", "") or ""

        opt0 = (r.get("answer0", "") or "").strip()
        opt1 = (r.get("answer1", "") or "").strip()
        opt2 = (r.get("answer2", "") or "").strip()
        opt3 = (r.get("answer3", "") or "").strip()

        options = {"A": opt0, "B": opt1, "C": opt2, "D": opt3}

        label_idx = get_label_index(r)
        if label_idx is None or label_idx not in IDX_TO_LETTER:
            # skip rows with bad/missing labels
            continue

        answer_letter = IDX_TO_LETTER[label_idx]

        item = {
            "id": rid,
            "type": "mcq",
            "stem": normalize_stem(context, question),
            "options": options,                 # IMPORTANT: your run_experiment uses "options"
            "answer": [answer_letter],          # list of letters, same as your bioc406 dataset
            "answer_format": "letters",
            "source": {
                "csv_path": args.in_csv,
                "row_index": args.start + i,
                "label_index": label_idx,
            },
        }
        items.append(item)

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for item in items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} items to: {args.out_jsonl}")

if __name__ == "__main__":
    main()
