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


