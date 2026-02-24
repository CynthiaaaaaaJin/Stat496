# from __future__ import annotations

# import argparse
# import json
# import time
# from typing import List

# from src.data_io import iter_dataset_items
# from src.prompts import build_prompt
# from src.parsing import parse_answer, is_correct
# from src.backends.chatgpt_backend import ChatGPTBackend


# def parse_csv_list(s: str) -> List[str]:
#     return [x.strip() for x in s.split(",") if x.strip()]


# def main() -> None:
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--model-name", required=True, help="OpenAI model name, e.g. gpt-5-mini, gpt-4.1-mini")
#     ap.add_argument("--rpm-limit", type=int, default=500, help="Requests per minute limit.")

#     ap.add_argument("--dataset", required=True, help="JSONL dataset path.")
#     ap.add_argument("--out-jsonl", default="outputs/runs_chatgpt.jsonl")

#     ap.add_argument("--treatments", nargs="+", default=["T0", "T5"])
#     ap.add_argument("--temps", default="0.2")
#     ap.add_argument("--k", type=int, default=1)

#     ap.add_argument("--max-tokens", type=int, default=256)
#     ap.add_argument("--top-p", type=float, default=0.95)
#     ap.add_argument("--repeat-penalty", type=float, default=1.0, help="Kept for compatibility (OpenAI may ignore).")
#     ap.add_argument("--seed", type=int, default=-1, help="If >=0, pass seed (model support may vary).")

#     ap.add_argument("--allow-explanation", action="store_true", help="Include explanation field content.")

#     # Optional: set OPENAI_BASE_URL for proxies, or pass through env.
#     ap.add_argument("--openai-base-url", default="", help="Optional base_url override (advanced).")

#     args = ap.parse_args()

#     treatments = args.treatments
#     temps = [float(x) for x in parse_csv_list(args.temps)]
#     k = args.k
#     seed = None if args.seed < 0 else args.seed

#     backend = ChatGPTBackend(
#         model_name=args.model_name,
#         rpm_limit=args.rpm_limit,
#         base_url=args.openai_base_url or None,
#     )

#     items = list(iter_dataset_items(args.dataset))
#     if not items:
#         raise ValueError(f"No items found in dataset: {args.dataset}")

#     import os
#     os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

#     with open(args.out_jsonl, "w", encoding="utf-8") as fout:
#         for t in treatments:
#             for temp in temps:
#                 config_id = f"{t}_temp{temp}"
#                 for item in items:
#                     qid = item["id"]
#                     stem = item["stem"]
#                     options = item.get("options", None)
#                     gt = item.get("answer", [])
#                     fmt = item.get("answer_format", "letters")

#                     for r in range(k):
#                         run_id = f"{config_id}__{qid}__r{r}"

#                         prompt = build_prompt(
#                             treatment=t,
#                             stem=stem,
#                             options=options,
#                             # keep your treatment logic intact; output contract handled by backend
#                             allow_explanation=args.allow_explanation and (t != "T1"),
#                         )

#                         t0 = time.time()
#                         res = backend.generate(
#                             prompt=prompt,
#                             temperature=temp,
#                             max_tokens=args.max_tokens,
#                             top_p=args.top_p,
#                             repeat_penalty=args.repeat_penalty,
#                             seed=seed,
#                             allow_explanation=args.allow_explanation and (t != "T1"),
#                         )
#                         latency = time.time() - t0

#                         parsed = parse_answer(item, res.text)
#                         correct = is_correct(parsed, gt, fmt)

#                         row = {
#                             "run_id": run_id,
#                             "config_id": config_id,
#                             "treatment": t,
#                             "temperature": temp,
#                             "k": k,
#                             "question_id": qid,
#                             "question_type": item.get("type", ""),
#                             "answer_format": fmt,
#                             "prompt": prompt,
#                             "raw_output": res.text,  # JSON string
#                             "parsed_answer": parsed,
#                             "ground_truth": gt,
#                             "correct": correct,
#                             "token_count_method": res.token_count_method,
#                             "input_tokens": res.input_tokens,
#                             "output_tokens": res.output_tokens,
#                             "latency_sec": latency,
#                             "model_name": args.model_name,
#                             "rpm_limit": args.rpm_limit,
#                             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#                         }
#                         fout.write(json.dumps(row, ensure_ascii=False) + "\n")

#     print(f"Wrote runs to: {args.out_jsonl}")


# if __name__ == "__main__":
#     main()
from __future__ import annotations

import argparse
import json
import time
from typing import List

from src.data_io import iter_dataset_items
from src.prompts import build_prompt
from src.parsing import parse_answer, is_correct
from src.backends.chatgpt_backend import ChatGPTBackend


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model-name", required=True, help="OpenAI model name, e.g. gpt-4.1-mini")
    ap.add_argument("--rpm-limit", type=int, default=60, help="Requests per minute limit.")
    ap.add_argument("--dataset", required=True, help="JSONL dataset path.")
    ap.add_argument("--out-jsonl", default="outputs/runs_chatgpt.jsonl")

    ap.add_argument("--treatments", nargs="+", default=["T0", "T5"])
    ap.add_argument("--temps", default="0.2")
    ap.add_argument("--k", type=int, default=1)

    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repeat-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=-1)

    ap.add_argument("--allow-explanation", action="store_true", help="Allow explanation before FINAL line.")

    args = ap.parse_args()

    treatments = args.treatments
    temps = [float(x) for x in parse_csv_list(args.temps)]
    k = args.k
    seed = None if args.seed < 0 else args.seed

    backend = ChatGPTBackend(model_name=args.model_name, rpm_limit=args.rpm_limit)

    items = list(iter_dataset_items(args.dataset))
    if not items:
        raise ValueError(f"No items found in dataset: {args.dataset}")

    import os
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
                            temperature=temp,
                            max_tokens=args.max_tokens,
                            top_p=args.top_p,
                            repeat_penalty=args.repeat_penalty,
                            seed=seed,
                        )
                        latency = time.time() - t0

                        parsed = parse_answer(item, res.text)
                        correct = is_correct(parsed, gt, fmt)

                        row = {
                            "run_id": run_id,
                            "config_id": config_id,
                            "treatment": t,
                            "temperature": temp,
                            "k": k,
                            "question_id": qid,
                            "question_type": item.get("type", ""),
                            "answer_format": fmt,
                            "prompt": prompt,
                            "raw_output": res.text,
                            "parsed_answer": parsed,
                            "ground_truth": gt,
                            "correct": correct,
                            "token_count_method": res.token_count_method,
                            "input_tokens": res.input_tokens,
                            "output_tokens": res.output_tokens,
                            "latency_sec": latency,
                            "model_name": args.model_name,
                            "rpm_limit": args.rpm_limit,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote runs to: {args.out_jsonl}")


if __name__ == "__main__":
    main()