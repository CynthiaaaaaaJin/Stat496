import json
import re
from pathlib import Path
from gpt4all import GPT4All
from prompts import treatment_prompt

DATA_PATH = "data/apchem_sample_10.jsonl"
OUT_PATH  = "outputs/rq1_small_gpt4all_outputs.jsonl"
MODEL_FILE = "mistral-7b-instruct-v0.1Q4_0.gguf"
TEMPS = [0.2, 1.0]
TREATMENTS = ["T0", "T1", "T2", "T3", "T4"]
REPEATS = 1
MAX_TOKENS = 256

FINAL_RE = re.compile(r"FINAL:\s*([A-E])", re.IGNORECASE)

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def parse_final_letter(text: str) -> str:
    if not text:
        return ""
    m = FINAL_RE.search(text)
    return m.group(1).upper() if m else ""

def main():
    Path("outputs").mkdir(exist_ok=True)

    questions = load_jsonl(DATA_PATH)
    print("Loaded questions:", len(questions))

    model = GPT4All(MODEL_FILE)
    print("Using GPT4All local model:", MODEL_FILE)

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        count = 0
        for q in questions:
            for tr in TREATMENTS:
                for temp in TEMPS:
                    for r in range(REPEATS):
                        prompt = treatment_prompt(tr, q)

                        with model.chat_session():
                            raw = model.generate(
                                prompt,
                                temp=temp,
                                max_tokens=MAX_TOKENS
                            )

                        pred = parse_final_letter(raw)

                        row = {
                            "qid": q["qid"],
                            "gold": q.get("answer", ""),
                            "treatment": tr,
                            "temperature": temp,
                            "repeat": r,
                            "pred": pred,
                            "raw_text": raw,
                            "usage": {},   # GPT4All doesn't provide token counts
                            "error": None
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                        count += 1
                        print(f"Saved {count}: Q{q['qid']} {tr} temp={temp} pred={pred}")

    print("Done. Output:", OUT_PATH)

if __name__ == "__main__":
    main()
