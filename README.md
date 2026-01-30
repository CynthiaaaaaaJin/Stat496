# STAT496 Small Test - RQ1

## Goal
Test how prompt instruction styles (T0–T4) and temperature affect:
- correctness (accuracy vs ground truth)
- stability (answer consistency across repeats)
- cost proxy (token usage, when available)

## Dataset
AP Chemistry multiple-choice sample (10 questions for small test).

## Treatments (Prompts)
- T0: normal
- T1: final only
- T2: short steps + final
- T3: evidence line quoting selected option + final
- T4: short steps + evidence + final

All prompts enforce the last line:
`Final:<LETTER>`

## Variables varied in small test
- treatment: T0–T4
- temperature: 0.2 vs 1.0
- repeats: 1 (small test), will increase later for stability

## How to run (Gemini free tier)
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="YOUR_KEY"
python src/run_rq1_small.py \
  --backend gemini \
  --model models/gemini-flash-latest \
  --data data/apchem_sample_10.jsonl \
  --out outputs/rq1_small_outputs.jsonl \
  --treatments T0 T1 T2 T3 T4 \
  --temps 0.2 1.0 \
  --repeats 1 \
  --rpm_limit 5
python src/analyze_rq1_small.py \
  --preds outputs/rq1_small_outputs.jsonl \
  --answers data/apchem_answers_sample_10.jsonl
