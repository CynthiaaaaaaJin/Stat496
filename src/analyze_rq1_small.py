import json
from collections import defaultdict, Counter

IN_PATH = "outputs/rq1_small_outputs.jsonl"

def main():
    rows = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        print("No rows found. Did the runner write outputs?")
        return

    # Accuracy by condition
    acc = defaultdict(lambda: {"n": 0, "correct": 0, "tokens": 0, "errors": 0})
    preds_by = defaultdict(list)  # (qid, treatment, temp) -> list of preds

    for r in rows:
        key = (r["treatment"], r["temperature"])
        acc[key]["n"] += 1
        if r.get("error"):
            acc[key]["errors"] += 1

        gold = (r.get("gold") or "").strip().upper()
        pred = (r.get("pred") or "").strip().upper()

        if gold and pred and gold == pred:
            acc[key]["correct"] += 1

        usage = r.get("usage") or {}
        # usage may be object-like; handle dict only
        if isinstance(usage, dict):
            total_tokens = usage.get("total_tokens") or usage.get("totalTokenCount") or 0
            acc[key]["tokens"] += int(total_tokens) if str(total_tokens).isdigit() else 0

        preds_by[(r["qid"], r["treatment"], r["temperature"])].append(pred)

    print("\n=== Accuracy / Cost summary by (treatment, temperature) ===")
    for key in sorted(acc.keys()):
        n = acc[key]["n"]
        correct = acc[key]["correct"]
        errors = acc[key]["errors"]
        tokens = acc[key]["tokens"]
        print(f"{key}: acc={correct}/{n}={correct/n:.2f} | errors={errors} | total_tokens={tokens}")

    # Stability (only meaningful if REPEATS >= 2)
    stable_n = 0
    total_groups = 0
    for k, preds in preds_by.items():
        # ignore empty predictions when checking stability
        preds_clean = [p for p in preds if p]
        if len(preds) <= 1:
            continue
        total_groups += 1
        if len(set(preds_clean)) == 1 and len(preds_clean) == len(preds):
            stable_n += 1

    print("\n=== Stability (requires repeats>=2) ===")
    if total_groups == 0:
        print("Not enough repeats to compute stability. Set REPEATS=2 in run script next.")
    else:
        print(f"Stable groups: {stable_n}/{total_groups} = {stable_n/total_groups:.2f}")

    # Variety check: distribution of predicted letters per condition
    print("\n=== Variety check: distribution of predicted letters per condition ===")
    for key in sorted(acc.keys()):
        letters = [r.get("pred","") for r in rows if (r["treatment"], r["temperature"]) == key]
        c = Counter([x for x in letters if x])
        print(f"{key}: {dict(c)}")

if __name__ == "__main__":
    main()
