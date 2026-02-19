from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple
import csv

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p + 1e-12, 2)
    return h

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-summary-csv", default="outputs/summary.csv")
    ap.add_argument("--out-per-question-csv", default="outputs/per_question.csv")
    args = ap.parse_args()

    rows = load_jsonl(args.in_jsonl)
    if not rows:
        raise ValueError("No rows found in input JSONL.")

    # group by (config_id, question_id)
    by_cq: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_config: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        by_cq[(r["config_id"], r["question_id"])].append(r)
        by_config[r["config_id"]].append(r)

    # per-question metrics
    perq = []
    for (cfg, qid), rr in by_cq.items():
        answers = [x.get("parsed_answer", "") for x in rr]
        counts = Counter(answers)
        mode, mode_ct = counts.most_common(1)[0] if answers else ("", 0)

        strict_stable = (len(counts) == 1 and mode != "")
        mode_freq = (mode_ct / len(rr)) if rr else 0.0
        ent = entropy_from_counts(counts)

        acc = sum(1 for x in rr if x.get("correct")) / max(1, len(rr))

        perq.append({
            "config_id": cfg,
            "question_id": qid,
            "k_runs": len(rr),
            "mode_answer": mode,
            "strict_stable": strict_stable,
            "mode_freq": round(mode_freq, 4),
            "answer_entropy_bits": round(ent, 4),
            "accuracy_over_runs": round(acc, 4),
        })

    # summary per config
    summary = []
    for cfg, rr in by_config.items():
        n = len(rr)
        num_correct = sum(1 for x in rr if x.get("correct"))
        acc = num_correct / max(1, n)

        # stability over questions: average of per-question strict stable and mode_freq
        qs = sorted(set(x["question_id"] for x in rr))
        strict_flags = []
        mode_freqs = []
        entropies = []
        for qid in qs:
            answers = [x.get("parsed_answer", "") for x in by_cq[(cfg, qid)]]
            counts = Counter(answers)
            mode, mode_ct = counts.most_common(1)[0] if answers else ("", 0)
            strict_flags.append(len(counts) == 1 and mode != "")
            mode_freqs.append(mode_ct / max(1, len(answers)))
            entropies.append(entropy_from_counts(counts))

        strict_stability = sum(strict_flags) / max(1, len(strict_flags))
        avg_mode_freq = sum(mode_freqs) / max(1, len(mode_freqs))
        avg_entropy = sum(entropies) / max(1, len(entropies))

        # token cost: use total tokens if present; otherwise blank
        totals = []
        for x in rr:
            it = x.get("input_tokens")
            ot = x.get("output_tokens")
            if isinstance(it, int) and isinstance(ot, int):
                totals.append(it + ot)
        avg_total_tokens = (sum(totals) / len(totals)) if totals else ""
        total_tokens = (sum(totals)) if totals else ""

        cost_per_correct = ""
        if totals and num_correct > 0:
            cost_per_correct = total_tokens / num_correct

        summary.append({
            "config_id": cfg,
            "n_runs": n,
            "n_correct": num_correct,
            "accuracy": round(acc, 4),
            "strict_stability": round(strict_stability, 4),
            "avg_mode_freq": round(avg_mode_freq, 4),
            "avg_entropy_bits": round(avg_entropy, 4),
        })

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.out_summary_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_per_question_csv) or ".", exist_ok=True)

    with open(args.out_per_question_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(perq[0].keys()))
        w.writeheader()
        w.writerows(perq)

    with open(args.out_summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"Wrote: {args.out_summary_csv}")
    print(f"Wrote: {args.out_per_question_csv}")

if __name__ == "__main__":
    main()
