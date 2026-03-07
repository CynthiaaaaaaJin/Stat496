#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

from src.data_io import iter_dataset_items
from src.prompts import build_prompt
from src.parsing import parse_answer, is_correct
from src.backends.chatgpt_backend import ChatGPTBackend


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model-name", required=True, help="OpenAI model name, e.g. gpt-4o-mini, gpt-4.1-mini")
    ap.add_argument("--rpm-limit", type=int, default=60, help="Requests per minute limit.")

    ap.add_argument("--dataset", required=True, help="JSONL dataset path.")
    ap.add_argument("--out-jsonl", default="outputs/runs_chatgpt.jsonl", help="Output JSONL path.")

    ap.add_argument("--treatments", nargs="+", default=["T0", "T5"], help="Treatments, e.g. T0 T1 ...")
    ap.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=[0.2],
        help="One or more temperatures, e.g. --temps 0.2 0.5 0.7",
    )
    ap.add_argument("--k", type=int, default=1, help="Repeats per (question, config).")

    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repeat-penalty", type=float, default=1.1, help="Kept for compatibility (OpenAI may ignore).")
    ap.add_argument("--seed", type=int, default=-1, help="If >=0, pass seed (model support may vary).")

    ap.add_argument("--allow-explanation", action="store_true", help="Allow explanation before FINAL line (prompt-side).")

    args = ap.parse_args()

    treatments = args.treatments
    temps = list(args.temps)  # list[float]
    k = int(args.k)
    seed: Optional[int] = None if args.seed < 0 else int(args.seed)

    backend = ChatGPTBackend(
        model_name=args.model_name,
        rpm_limit=args.rpm_limit,
    )

    items = list(iter_dataset_items(args.dataset))
    if not items:
        raise ValueError(f"No items found in dataset: {args.dataset}")

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for t in treatments:
            for temp in temps:
                config_id = f"{t}_temp{temp}"
                for item in items:
                    qid = item["id"]
                    stem = item["stem"]
                    options = item.get("options", None)
                    gt = item.get("answer", [])
                    fmt = item.get("answer_format", "letters")

                    for r in range(k):
                        run_id = f"{config_id}__{qid}__r{r}"

                        prompt = build_prompt(
                            treatment=t,
                            stem=stem,
                            options=options,
                            allow_explanation=args.allow_explanation and (t != "T1"),
                        )

                        t0 = time.time()
                        res = backend.generate(
                            prompt=prompt,
                            temperature=float(temp),
                            max_tokens=int(args.max_tokens),
                            top_p=float(args.top_p),
                            repeat_penalty=float(args.repeat_penalty),
                            seed=seed,
                        )
                        latency = time.time() - t0

                        parsed = parse_answer(item, res.text)
                        correct = is_correct(parsed, gt, fmt)

                        row = {
                            "run_id": run_id,
                            "config_id": config_id,
                            "treatment": t,
                            "temperature": float(temp),
                            "k": k,
                            "question_id": qid,
                            "question_type": item.get("type", ""),
                            "answer_format": fmt,
                            "prompt": prompt,
                            "raw_output": res.text,
                            "parsed_answer": parsed,
                            "ground_truth": gt,
                            "correct": bool(correct),
                            "token_count_method": getattr(res, "token_count_method", "openai"),
                            "input_tokens": getattr(res, "input_tokens", None),
                            "output_tokens": getattr(res, "output_tokens", None),
                            "latency_sec": float(latency),
                            "model_name": args.model_name,
                            "rpm_limit": int(args.rpm_limit),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote runs to: {args.out_jsonl}")


if __name__ == "__main__":
    main()