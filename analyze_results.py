"""
Analyze outputs.jsonl -> accuracy + stability.

- Accuracy: parsed_answer in ground_truth_answers
- Stability: within each (question_id, treatment, temperature), whether all K runs match

Usage:
  python analyze_results.py \
    --dataset bioc406_exam1A_subset/dataset.jsonl \
    --outputs bioc406_exam1A_subset/outputs.jsonl \
    --report bioc406_exam1A_subset/report.json
"""

import argparse, json
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    # ground truth
    gt = {}
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            gt[item["id"]] = set(item["answer"])

    # model runs
    runs = []
    with open(args.outputs, "r", encoding="utf-8") as f:
        for line in f:
            runs.append(json.loads(line))

    group = defaultdict(list)
    for r in runs:
        key = (r["question_id"], r["treatment"], r["temperature"])
        group[key].append(r)

    by_question = []
    for (qid, treatment, temp), rs in group.items():
        correct = [1 if (r.get("parsed_answer") in gt.get(qid, set())) else 0 for r in rs]
        answers = [r.get("parsed_answer") for r in rs]
        stable = int(len(set(answers)) == 1 and answers[0] is not None)

        by_question.append({
            "question_id": qid,
            "treatment": treatment,
            "temperature": temp,
            "k": len(rs),
            "accuracy": sum(correct) / max(1, len(correct)),
            "stable": stable,
            "answers": answers,
        })

    # aggregate by (treatment, temp)
    agg = defaultdict(list)
    for s in by_question:
        agg[(s["treatment"], s["temperature"])].append(s)

    by_config = []
    for (treat, temp), items in agg.items():
        by_config.append({
            "treatment": treat,
            "temperature": temp,
            "n_questions": len(items),
            "mean_accuracy": sum(x["accuracy"] for x in items) / len(items),
            "stability_rate": sum(x["stable"] for x in items) / len(items),
        })

    report = {"by_question": by_question, "by_config": by_config}

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote report to {args.report}")

if __name__ == "__main__":
    main()
