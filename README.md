# STAT496 Small Test - RQ1(check write_up.txt)

## Goal
Test how prompt instruction styles (T0–T5) and temperature affect:
- correctness (accuracy vs ground truth)
- stability (answer consistency across repeats)
- cost proxy (token usage, when available)

## Dataset
Biochemistry 406 Exam - multiple-choice sample (10 questions for small test).

## Treatments (Prompts)
- T0: normal
- T1: final only
- T2: short steps + final
- T3: evidence line quoting selected option + final
- T4: short steps + evidence + final
- T5: normal + self check

All prompts enforce the last line:
`Final:<LETTER>`

## Variables varied in small test
- treatment: T0–T5
- temperature: 0.2 vs 0.7 vs 1.0
- repeats: 3 (small test), will increase later for stability

## Terminal Process:
python -m src.run_experiment \
  --model-filename "/Users/cynthiajyx/Library/Application Support/nomic.ai/GPT4All/gpt4all-falcon-newbpe-q4_0.gguf" \
  --dataset data/blog_10.jsonl \
  --out-jsonl outputs/runs_blog10_T0_T5_t02_t07_k3_max256.jsonl \
  --treatments T0 T1 T2 T3 T4 T5 \
  --temps 0.2,0.7 \
  --k 3 \
  --max-tokens 256 \
  --allow-explanation

python -m src.analyze_results \
--in-jsonl outputs/runs_blog10_T0_T5_t02_t07_k3_max256.jsonl \
 --out-summary-csv outputs/summary_blog10.csv \
--out-per-question-csv outputs/per_question_blog10.csv

