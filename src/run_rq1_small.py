import os, json, time, re
from pathlib import Path
from google import genai
from prompts import treatment_prompt

# ======= FREE TIER SAFETY SETTINGS =======
# Your error showed limit: 5 requests/min => 1 request per ~12s
SECONDS_BETWEEN_CALLS = 13
MAX_RETRIES_429 = 5
RETRY_SLEEP_SECONDS = 50

MODEL = "models/gemini-flash-latest"  # stable & cheap on free tier
# If you want: "models/gemini-3-flash-preview" (but same rate limit usually)
# MODEL = "models/gemini-3-flash-preview"

TREATMENTS = ["T0", "T1", "T2", "T3", "T4"]  # add T5 later if you want
TEMPS = [0.2, 0.8]
REPEATS = 1  # set to 2 later for stability (will double calls)
SECONDS_BETWEEN_CALLS = 13
DATA_PATH = "data/apchem_sample_10.jsonl"
OUT_PATH = "outputs/rq1_small_outputs.jsonl"
MODEL = "models/gemini-flash-latest"
FINAL_RE = re.compile(r"FINAL:\s*([A-E])", re.IGNORECASE)

def load_jsonl(path):
    import json
    items=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                items.append(json.loads(line))
    return items


def parse_final_letter(text: str) -> str:
    if not text:
        return ""
    m = FINAL_RE.search(text)
    return m.group(1).upper() if m else ""

def call_model(client, prompt: str, temperature: float):
    # Basic generate call. Add max_output_tokens to keep cost lower.
    return client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "temperature": temperature,
            "max_output_tokens": 256,
        }
    )

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY. Set it with export GEMINI_API_KEY=...")

    Path("outputs").mkdir(exist_ok=True)

    questions = load_jsonl(DATA_PATH)
    print("Loaded questions =", len(questions))
    print("Model =", MODEL)
    print("Writing to =", OUT_PATH)

    client = genai.Client(api_key=api_key)

    total_calls = len(questions) * len(TREATMENTS) * len(TEMPS) * REPEATS
    print("Planned calls =", total_calls, "(free-tier safe but will take time due to throttle)")

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        call_count = 0
        for q in questions:
            for tr in TREATMENTS:
                for temp in TEMPS:
                    for r in range(REPEATS):
                        prompt = treatment_prompt(tr, q)

                        raw_text = ""
                        usage = None
                        err = None

                        # retry loop (handles 429)
                        for attempt in range(MAX_RETRIES_429):
                            try:
                                resp = call_model(client, prompt, temp)
                                raw_text = getattr(resp, "text", "") or ""
                                usage_obj = getattr(resp, "usage_metadata", None)
                                usage = {}
                                if usage_obj is not None:
                                    # make it JSON-serializable
                                    usage = {
                                        "prompt_tokens": getattr(usage_obj, "prompt_token_count", None),
                                        "output_tokens": getattr(usage_obj, "candidates_token_count", None),
                                        "total_tokens": getattr(usage_obj, "total_token_count", None),
                                    }
                                err = None
                                break
                            except Exception as e:
                                msg = str(e)
                                # 429 quota/rate limit
                                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                                    err = msg
                                    print(f"[429] sleeping {RETRY_SLEEP_SECONDS}s then retry {attempt+1}/{MAX_RETRIES_429}")
                                    time.sleep(RETRY_SLEEP_SECONDS)
                                    continue
                                else:
                                    err = msg
                                    break

                        pred = parse_final_letter(raw_text)

                        row = {
                            "qid": q["qid"],
                            "gold": q.get("answer", ""),
                            "treatment": tr,
                            "temperature": temp,
                            "repeat": r,
                            "pred": pred,
                            "raw_text": raw_text,
                            "usage": usage if usage is not None else {},
                            "error": err
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()

                        call_count += 1
                        print(f"Saved {call_count}/{total_calls}: Q{q['qid']} {tr} temp={temp} r={r} pred={pred} err={('yes' if err else 'no')}")

                        # throttle to stay within free tier rate limit
                        time.sleep(SECONDS_BETWEEN_CALLS)

    print("Done. Output saved to:", OUT_PATH)

if __name__ == "__main__":
    main()
