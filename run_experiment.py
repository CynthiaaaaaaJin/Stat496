"""
Run prompt treatments on a JSONL dataset using GPT4All.

Usage example:
  python run_experiment.py \
    --dataset bioc406_exam1A_subset/dataset.jsonl \
    --model_name deepseek-...gguf \
    --model_path /path/to/models \
    --out bioc406_exam1A_subset/outputs.jsonl \
    --temps 0.0 0.7 \
    --k 3

Requires:
  pip install gpt4all
"""

import argparse, json, re, time
from typing import Dict, Any, Optional

try:
    from gpt4all import GPT4All
except ImportError as e:
    raise SystemExit("Please `pip install gpt4all` first.") from e

LETTER_RE = re.compile(r"\b([A-E])\b")

TREATMENTS = {
    "T0_normal": "Answer the multiple-choice question.",
    "T1_final_only": "Only output the final answer letter (A, B, C, D, or E). No explanation.",
    "T2_step_by_step": "Think step by step, then output the final answer letter on a new line as `Final: X`.",
    "T3_use_options_as_background": "Use ONLY the provided options and the question statement as your background. Choose the best answer. Output just the letter.",
    "T4_step_by_step_with_background": "Use the options as your only background. Reason step by step, then output `Final: X`.",
    "T5_self_check": "Choose an answer, then do a quick self-check against each option, and output `Final: X`."
}

def build_prompt(item, treatment_name: str) -> str:
    choices = "\n".join([f"{k}. {v}" for k, v in item["choices"].items()])
    instruction = TREATMENTS[treatment_name]
    return (
        f"{instruction}\n\n"
        f"Question:\n{item['question']}\n\n"
        f"Options:\n{choices}\n\n"
        "IMPORTANT: Output exactly ONE line in this format:\n"
        "Final: <A|B|C|D|E>\n"
        "Do not write anything else.\n"
    )


def parse_answer(text: str):
    if not text:
        return None
    t = text.strip().upper()

    # 1) strongest: "Final: X"
    m = re.search(r"\bFINAL\b\s*[:\-]\s*([A-E])\b", t)
    if m:
        return m.group(1)

    # 2) "Answer: X", "Option X", "Choice X"
    m = re.search(r"\b(ANSWER|OPTION|CHOICE)\b\s*[:\-]?\s*([A-E])\b", t)
    if m:
        return m.group(2)

    # 3) "(X)" or "[X]" or "{X}"
    m = re.search(r"[\(\[\{]\s*([A-E])\s*[\)\]\}]", t)
    if m:
        return m.group(1)

    # 4) a line like "C." or "C)" (often models do this)
    m = re.search(r"(?m)^\s*([A-E])\s*[\.\)]\s*$", t)
    if m:
        return m.group(1)

    # 5) "A." anywhere (but be careful: may appear in option list echoes)
    m = re.search(r"\b([A-E])\.", t)
    if m:
        return m.group(1)

    # 6) last resort: any standalone letter token
    m = re.search(r"\b([A-E])\b", t)
    return m.group(1) if m else None



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.7])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    model = GPT4All(args.model_name, model_path=args.model_path)

    items = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    with open(args.out, "w", encoding="utf-8") as outf:
        for item in items:
            for treatment in TREATMENTS.keys():
                prompt = build_prompt(item, treatment)
                for temp in args.temps:
                    for run_idx in range(args.k):
                        t0 = time.time()
                        raw = model.generate(prompt, temp=temp, max_tokens=args.max_tokens)
                        dt = time.time() - t0
                        parsed = parse_answer(raw or "")
                        row = {
                            "question_id": item["id"],
                            "treatment": treatment,
                            "temperature": temp,
                            "run_index": run_idx,
                            "prompt": prompt,
                            "raw_output": raw,
                            "parsed_answer": parsed,
                            "runtime_s": dt,
                        }
                        outf.write(json.dumps(row, ensure_ascii=False) + "\n")
                        outf.flush()

    print(f"Done. Wrote outputs to {args.out}")

if __name__ == "__main__":
    main()
