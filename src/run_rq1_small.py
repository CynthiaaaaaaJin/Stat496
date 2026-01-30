import argparse
import json
import os
import re
from typing import Dict, Any, List

from tqdm import tqdm

from prompts import treatment_prompt
from backends import GeminiBackend, GPT4AllBackend


FINAL_RE = re.compile(r"Final:\s*([A-E])", re.IGNORECASE)


def parse_final_letter(text: str) -> str:
    if not text:
        return ""
    m = FINAL_RE.search(text)
    return (m.group(1).upper() if m else "")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gemini", "gpt4all"], default="gemini")
    ap.add_argument("--model", type=str, default="models/gemini-flash-latest",
                    help="Gemini: model name like models/gemini-flash-latest | GPT4All: path to .gguf")
    ap.add_argument("--data", type=str, default="data/apchem_sample_10.jsonl")
    ap.add_argument("--out", type=str, default="outputs/rq1_small_outputs.jsonl")
    ap.add_argument("--treatments", nargs="+", default=["T0", "T1", "T2", "T3", "T4"])
    ap.add_argument("--temps", nargs="+", type=float, default=[0.2, 1.0])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max_output_tokens", type=int, default=128)
    ap.add_argument("--rpm_limit", type=int, default=5, help="Gemini free tier safety limit")
    args = ap.parse_args()

    questions = read_jsonl(args.data)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.backend == "gemini":
        backend = GeminiBackend(model_name=args.model, rpm_limit=args.rpm_limit)
    else:
        backend = GPT4AllBackend(model_path=args.model)

    planned = len(questions) * len(args.treatments) * len(args.temps) * args.repeats
    print(f"Backend={args.backend} | Model={args.model}")
    print(f"Writing to={args.out}")
    print(f"Planned calls={planned}")

    with open(args.out, "a", encoding="utf-8") as out:
        for q in tqdm(questions, desc="Running"):
            for treatment in args.treatments:
                prompt = treatment_prompt(treatment, q)
                for temp in args.temps:
                    for r in range(args.repeats):
                        try:
                            resp = backend.generate(
                                prompt=prompt,
                                temperature=temp,
                                max_output_tokens=args.max_output_tokens,
                            )
                            raw = resp.get("raw_text", "")
                            pred = parse_final_letter(raw)
                            usage = resp.get("usage", {}) or {}

                            row = {
                                "qid": q["qid"],
                                "treatment": treatment,
                                "temperature": temp,
                                "repeat": r,
                                "pred": pred,
                                "raw_text": raw,
                                "usage": usage,
                                "model": resp.get("model", args.model),
                            }
                        except Exception as e:
                            row = {
                                "qid": q["qid"],
                                "treatment": treatment,
                                "temperature": temp,
                                "repeat": r,
                                "pred": "",
                                "raw_text": "",
                                "usage": {},
                                "model": args.model,
                                "error": str(e),
                            }

                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()


if __name__ == "__main__":
    main()
