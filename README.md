# STAT496 – Small Test (RQ1 Pilot)

## Goal
Test whether different prompt instruction styles change correctness and token cost on a small multiple-choice dataset (AP Chemistry).

## Data
- `data/apchem_small.jsonl` (2–3 AP Chem MC questions, manually entered)
- Ground-truth answers included as `"answer"` in the JSONL.

## Treatments (Prompts)
- T0 Normal answering
- T1 Final answer only
- T2 Step-by-step (still return FINAL only)
- T3 Cite background (quote the selected choice text)
- T4 Step-by-step + cite background

## Variables varied in this pilot
- temperature ∈ {0.2, 0.8}
- prompt treatment ∈ {T0..T4}
- repeats = 1 (will expand to repeats=2 for stability)

## How we ran it
- Model: `models/gemini-flash-latest` (free-tier friendly)
- Throttle: 1 request every 13 seconds to avoid 429 free-tier rate limit
- Output saved to: `outputs/rq1_small_outputs.jsonl`

## Example prompts
(Include 1–2 prompts here, e.g., T0 and T4)

## What responses we received (examples)
(Include 2–3 interesting outputs: correct, wrong, maybe a temp=0.8 weird one)

## Initial results
Paste output from:
`python src/analyze_rq1_small.py`

## How might we improve / expand?
- Increase dataset size (e.g., 20–60 questions)
- Add repeats=2 to quantify stability
- Add T5 self-check
- Compare across models: flash vs pro vs gemini-3
- Automate data entry (later): parse PDFs or use a clean public MC dataset
- Automate large-scale collection with batching + rate-limit aware scheduling
