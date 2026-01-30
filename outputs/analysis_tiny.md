# STAT 496 — Small Scale Prompt × Temperature Experiment

## Goal

Check how different prompts (instruction styles) and temperature influence correctness and stability for a small set of multiple-choice questions.

## Data

- Source: Biochem 406 Exam 1 (Version A), selected questions: 6,8,9,10,11,12,13,14,15,17.

- Ground truth answers stored in `data/answers_clean.json`.

## Experimental Variables

- Prompt treatments: T0–T5

- Temperature: as configured in the run

- Repeats per (config, question): K

## Metrics

- Accuracy: correct / total

- Stability (per question): all K runs produce the same parsed final choice

- Token cost: N × avg_tokens_per_run, where avg_tokens_per_run = (input_tokens + output_tokens)/2

## Results Summary (per config)

| config_id   |   n_runs |   accuracy |   stability |   avg_tokens_per_run |
|:------------|---------:|-----------:|------------:|---------------------:|
| T0_temp0.2  |       10 |      0.5   |        1    |              88.4    |
| T5_temp0.2  |        8 |      0.125 |        0.25 |              65.8125 |


## Per-question Stability Snapshot

| config_id   |   question_id |   k_runs | answers_mode   | stable   |   accuracy_over_runs |
|:------------|--------------:|---------:|:---------------|:---------|---------------------:|
| T0_temp0.2  |             6 |        1 | C              | True     |                    0 |
| T0_temp0.2  |             8 |        1 | D              | True     |                    1 |
| T0_temp0.2  |             9 |        1 | B              | True     |                    1 |
| T0_temp0.2  |            10 |        1 | B              | True     |                    0 |
| T0_temp0.2  |            11 |        1 | A              | True     |                    0 |
| T0_temp0.2  |            12 |        1 | A              | True     |                    1 |
| T0_temp0.2  |            13 |        1 | E              | True     |                    1 |
| T0_temp0.2  |            14 |        1 | C              | True     |                    1 |
| T0_temp0.2  |            15 |        1 | E              | True     |                    0 |
| T0_temp0.2  |            17 |        1 | C              | True     |                    0 |
| T5_temp0.2  |             6 |        1 | C              | True     |                    0 |
| T5_temp0.2  |             8 |        1 | D              | True     |                    1 |
| T5_temp0.2  |             9 |        1 | nan            | False    |                    0 |
| T5_temp0.2  |            10 |        1 | nan            | False    |                    0 |
| T5_temp0.2  |            11 |        1 | nan            | False    |                    0 |
| T5_temp0.2  |            12 |        1 | nan            | False    |                    0 |
| T5_temp0.2  |            13 |        1 | nan            | False    |                    0 |
| T5_temp0.2  |            14 |        1 | nan            | False    |                    0 |


## Example Model Outputs (one sample per config × question)

### T0_temp0.2 — Q10

- Parsed answer: `B` | Correct: `False`

Prompt (truncated):

```text
Hartnup disease is a genetic syndrome in which there is an inactivating mutation in a gene required for the absorption of tryptophan. Based on this information, which of these drugs might help treat the neurological symptoms of patients with Hartnup disease?

A. A selective serotonin reuptake inhibitor (SSRI).
B. A dopamine reuptake inhibitor.
C. An antihistamine.
D. Reserpine, which depletes monoamines from synaptic vesicles.
E. A GPCR inhibitor. Biochemistry 406 Jan. 26, 2026 Exam 1  -  Version A Page 5

Answer the question.

You MUST follow this format strictly:
First line: FINAL: <A/B/C/D/ ...[truncated]
```

Output (truncated):

```text
FINAL: B The Hartnup disease is a genetic disorder in which there is an inability to absorb tryptophan. Tryptophan is a precursor for serotonin and dopamine. Therefore, drugs that increase levels of these neurotransmitters would be beneficial for patients with this disease. A selective serotonin reuptake inhibitor (SSRI) increases the amount of serotonin available for synaptic transmission, but it ...[truncated]
```

### T0_temp0.2 — Q11

- Parsed answer: `A` | Correct: `False`

Prompt (truncated):

```text
Biotin is attached to carboxylase enzymes through the action of a biotin-ligating protein. Which of these would occur if you inhibited the activity of the biotin-ligating protein in rodents?

A. Fatty acid oxidation would be inhibited.
B. Acetyl-CoA would accumulate in the cytoplasm.
C. Palmitate would accumulate.
D. Malonyl CoA would accumulate.
E. TAG synthesis would increase.

Answer the question.

You MUST follow this format strictly:
First line: FINAL: <A/B/C/D/E>
After that, you may add explanation.
Do NOT change the word FINAL.


```

Output (truncated):

```text
FINAL: A

The biotin-ligating protein attaches biotin to carboxylase enzymes. Carboxylases are involved in fatty acid synthesis and oxidation. If the activity of this protein is inhibited, then biotin will not be attached to these enzymes, and as a result, they will not be able to function properly. This means that fatty acid oxidation would be inhibited because the carboxylase enzyme required for ...[truncated]
```

### T0_temp0.2 — Q12

- Parsed answer: `A` | Correct: `True`

Prompt (truncated):

```text
Folate (vitamin B9) deficiency can lead to megaloblastic anemia, where red blood cells become abnormally large. You see a patient with severe megaloblastic anemia even though they have eaten spinach (a good source of folate) daily for several weeks. Which of these could explain this observation?

A. A defect in dihydrofolate reductase (DHFR).
B. Accumulation of S-adenosylmethionine.
C. Decreased expression of intrinsic factor.
D. Increased production of glutathione.
E. Decreased activity of pancreatic lipase.

Answer the question.

You MUST follow this format strictly:
First line: FINAL: <A/B/ ...[truncated]
```

Output (truncated):

```text
FINAL: A

The patient's daily consumption of spinach would normally provide sufficient folate to prevent megaloblastic anemia. However, a defect in dihydrofolate reductase (DHFR) would impair the conversion of dihydrofolate to tetrahydrofolate, which is necessary for DNA synthesis and repair. This impairment would lead to a functional folate deficiency despite adequate dietary intake.

Explanation ...[truncated]
```

### T0_temp0.2 — Q13

- Parsed answer: `E` | Correct: `True`

Prompt (truncated):

```text
Which of these fatty acids is synthesized using BOTH a desaturase AND an elongase in plants?

A. C18:1(D9)
B. C18:2(D6, 9)
C. C20:0
D. C20:1(D9)
E. C20:2(D6, 9)

Answer the question.

You MUST follow this format strictly:
First line: FINAL: <A/B/C/D/E>
After that, you may add explanation.
Do NOT change the word FINAL.


```

Output (truncated):

```text
FINAL: E

The correct answer is C20:2(D6, 9), which requires both a desaturase and an elongase to be synthesized in plants. This fatty acid has two double bonds (D6 and D9) and is longer than 18 carbons, indicating that it was produced using an elongase. The presence of two double bonds also suggests the involvement of a desaturase. Therefore, option E is the correct choice.
```

### T0_temp0.2 — Q14

- Parsed answer: `C` | Correct: `True`

Prompt (truncated):

```text
Which of these is NOT a source for amino acid carbon skeletons in mammals?

A. Glycolysis
B. Pyruvate
C. PPP (pentose phosphate pathway)
D. TCA cycle
E. All of these can be used to generate carbon skeletons. Biochemistry 406 Jan. 26, 2026 Exam 1  -  Version A Page 6

Answer the question.

You MUST follow this format strictly:
First line: FINAL: <A/B/C/D/E>
After that, you may add explanation.
Do NOT change the word FINAL.


```

Output (truncated):

```text
FINAL: C

The pentose phosphate pathway (PPP) is primarily involved in generating NADPH and ribose-5-phosphate for nucleotide synthesis. While it does produce some intermediates that can be used to generate amino acid carbon skeletons, such as glucose-6-phosphate and fructose-6-phosphate, the PPP itself is not a primary source of carbon skeletons for amino acid synthesis in mammals.

The other opt ...[truncated]
```

### T0_temp0.2 — Q15

- Parsed answer: `E` | Correct: `False`

Prompt (truncated):

```text
Which of these statements about biological nitrogen fixing is TRUE?

A. The reduction of N2 to NH3 occurs at the FeMo cofactor.
B. The P-cluster dissociates after each electron is transferred to the FeMo cofactor.
C. ATP hydrolysis stimulates the reduction of N2 to NH3.
D. The reduction of N2 to NH3 occurs when 6 electrons have collected at the FeMo cofactor.
E. Reduction of the dinitrogen reductase (Fe-protein) requires ATP hydrolysis.

Answer the question.

You MUST follow this format strictly:
First line: FINAL: <A/B/C/D/E>
After that, you may add explanation.
Do NOT change the word FINAL.
 ...[truncated]
```

Output (truncated):

```text
FINAL: E

The reduction of N2 to NH3 occurs in two steps. First, the Fe-protein (dinitrogen reductase) is reduced by ATP hydrolysis, and then this reduced form donates electrons to the MoFe-protein (nitrogenase), which reduces N2 to NH3. So, the correct answer is E. The reduction of the dinitrogen reductase (Fe-protein) requires ATP hydrolysis. 

Note: The other options are incorrect because A ref ...[truncated]
```

### T0_temp0.2 — Q17

- Parsed answer: `C` | Correct: `False`

Prompt (truncated):

```text
What would you expect to happen to cellular lipid levels if you add a drug that increases the activity of phosphatidic acid phosphatase (lipin)?

A. Increased accumulation of triacylglycerol.
B. Increased production of membrane phospholipids.
C. Increased amount of phosphatidic acid.
D. Increased production of sphingolipids.
E. Increased synthesis of cholesterol. Biochemistry 406 Jan. 26, 2026 Exam 1  -  Version A Page 7

Answer the question.

You MUST follow this format strictly:
First line: FINAL: <A/B/C/D/E>
After that, you may add explanation.
Do NOT change the word FINAL.


```

Output (truncated):

```text
FINAL: C

The correct answer is phosphatidic acid because lipin (phosphatidic acid phosphatase) catalyzes the conversion of phosphatidic acid to diacylglycerol. If this enzyme's activity increases, it would lead to an increase in the production of phosphatidic acid as a substrate for the reaction.

Note: This question requires knowledge of lipid metabolism and the specific role of lipin in convert ...[truncated]
```

### T0_temp0.2 — Q6

- Parsed answer: `C` | Correct: `False`

Prompt (truncated):

```text
In 1982, a 42-year-old patient was hospitalized in San Jose, frozen like a statue. Then two drug-addict brothers in their 20s were reported to have advanced symptoms of Parkinson’s. All these patients had consumed the same illegal street drug, which turned out to have an impurity. The patients were given L-DOPA, which alleviated their motor symptoms. Based on these data, which of these neurotransmission pathways can you infer were NOT disrupted by the impurity?

A. Norepinephrine
B. Serotonin
C. Dopamine
D. Epinephrine
E. All of these have been altered.

Answer the question.

You MUST follow t ...[truncated]
```

Output (truncated):

```text
FINAL: C

The impurity was found to be MPTP (1-methyl-4-phenyl-1,2,5,6-tetrahydropyridine), which is a neurotoxin that selectively destroys dopaminergic neurons. The patients who consumed this drug experienced severe Parkinson's-like symptoms, including tremors, rigidity, and bradykinesia (slow movement). Since L-DOPA was effective in alleviating their motor symptoms, it suggests that the disrupti ...[truncated]
```

### T0_temp0.2 — Q8

- Parsed answer: `D` | Correct: `True`

Prompt (truncated):

```text
Free ammonia is captured by synthesis of   (1)  , an amino acid, and then transferred to the amino acid   (2)  . This amino acid (2) provides amino groups for amino acid synthesis.

A. 1=glutamate; 2=glutamine.
B. 1=glutamate; 2=glutamate.
C. 1=glutamine; 2=glutamine.
D. 1=glutamine; 2=glutamate.
E. Ammonia cannot be incorporated into amino acids.

Answer the question.

You MUST follow this format strictly:
First line: FINAL: <A/B/C/D/E>
After that, you may add explanation.
Do NOT change the word FINAL.


```

Output (truncated):

```text
FINAL: D

The correct answer is glutamine (1) and glutamate (2). Glutamine captures free ammonia through a reaction catalyzed by glutamine synthetase. The resulting glutamine molecule then donates its amino group to form glutamate, which can be used for the synthesis of other amino acids. This process effectively transfers the nitrogen from ammonia into an amino acid, making it available for furth ...[truncated]
```

### T0_temp0.2 — Q9

- Parsed answer: `B` | Correct: `True`

Prompt (truncated):

```text
You keep a pet rat, Poppy, in your room. You come home after a long day, and Poppy is hungry – he hasn’t eaten in almost 12 hours. When you open the door to the cabinet where you store the rat food, your cat jumps out lunges toward Poppy. Poppy is scared and has clearly activated the “fight or flight” response, and his blood glucose levels rise rapidly. Which combination of molecular responses has been activated in Poppy’s tissues?

A. ↑ AMPK activity, ↓ mTOR activity, ↓ fructose-2,6-bisphosphate, ↑ glycogen synthase activity
B. ↑ cAMP, ↑ PKA activity, ↓ fructose-2,6-bisphosphate, ↑ gluconeoge ...[truncated]
```

Output (truncated):

```text
FINAL: B

The correct answer is B because when Poppy's blood glucose levels rise rapidly due to stress (fight or flight response), his body activates a series of molecular responses to manage this sudden increase in energy. The combination described in option B accurately reflects these changes:

- ↑ cAMP: Cyclic AMP (cAMP) levels increase as part of the cellular response to stress, leading to inc ...[truncated]
```

### T5_temp0.2 — Q10

- Parsed answer: `` | Correct: `False`

Prompt (truncated):

```text
Hartnup disease is a genetic syndrome in which there is an inactivating mutation in a gene required for the absorption of tryptophan. Based on this information, which of these drugs might help treat the neurological symptoms of patients with Hartnup disease?

A. A selective serotonin reuptake inhibitor (SSRI).
B. A dopamine reuptake inhibitor.
C. An antihistamine.
D. Reserpine, which depletes monoamines from synaptic vesicles.
E. A GPCR inhibitor. Biochemistry 406 Jan. 26, 2026 Exam 1  -  Version A Page 5

Pick the best option.
Then do a quick self-check: try to find a reason your choice could ...[truncated]
```

Output (truncated):

```text

```

### T5_temp0.2 — Q11

- Parsed answer: `` | Correct: `False`

Prompt (truncated):

```text
Biotin is attached to carboxylase enzymes through the action of a biotin-ligating protein. Which of these would occur if you inhibited the activity of the biotin-ligating protein in rodents?

A. Fatty acid oxidation would be inhibited.
B. Acetyl-CoA would accumulate in the cytoplasm.
C. Palmitate would accumulate.
D. Malonyl CoA would accumulate.
E. TAG synthesis would increase.

Pick the best option.
Then do a quick self-check: try to find a reason your choice could be wrong.
Finally commit to ONE letter.
Do NOT show the self-check.

You MUST follow this format strictly:
First line: FINAL: <A ...[truncated]
```

Output (truncated):

```text

```

### T5_temp0.2 — Q12

- Parsed answer: `` | Correct: `False`

Prompt (truncated):

```text
Folate (vitamin B9) deficiency can lead to megaloblastic anemia, where red blood cells become abnormally large. You see a patient with severe megaloblastic anemia even though they have eaten spinach (a good source of folate) daily for several weeks. Which of these could explain this observation?

A. A defect in dihydrofolate reductase (DHFR).
B. Accumulation of S-adenosylmethionine.
C. Decreased expression of intrinsic factor.
D. Increased production of glutathione.
E. Decreased activity of pancreatic lipase.

Pick the best option.
Then do a quick self-check: try to find a reason your choice c ...[truncated]
```

Output (truncated):

```text

```

### T5_temp0.2 — Q13

- Parsed answer: `` | Correct: `False`

Prompt (truncated):

```text
Which of these fatty acids is synthesized using BOTH a desaturase AND an elongase in plants?

A. C18:1(D9)
B. C18:2(D6, 9)
C. C20:0
D. C20:1(D9)
E. C20:2(D6, 9)

Pick the best option.
Then do a quick self-check: try to find a reason your choice could be wrong.
Finally commit to ONE letter.
Do NOT show the self-check.

You MUST follow this format strictly:
First line: FINAL: <A/B/C/D/E>
After that, you may add explanation.
Do NOT change the word FINAL.


```

Output (truncated):

```text

```

### T5_temp0.2 — Q14

- Parsed answer: `` | Correct: `False`

Prompt (truncated):

```text
Which of these is NOT a source for amino acid carbon skeletons in mammals?

A. Glycolysis
B. Pyruvate
C. PPP (pentose phosphate pathway)
D. TCA cycle
E. All of these can be used to generate carbon skeletons. Biochemistry 406 Jan. 26, 2026 Exam 1  -  Version A Page 6

Pick the best option.
Then do a quick self-check: try to find a reason your choice could be wrong.
Finally commit to ONE letter.
Do NOT show the self-check.

You MUST follow this format strictly:
First line: FINAL: <A/B/C/D/E>
After that, you may add explanation.
Do NOT change the word FINAL.


```

Output (truncated):

```text

```

### T5_temp0.2 — Q6

- Parsed answer: `C` | Correct: `False`

Prompt (truncated):

```text
In 1982, a 42-year-old patient was hospitalized in San Jose, frozen like a statue. Then two drug-addict brothers in their 20s were reported to have advanced symptoms of Parkinson’s. All these patients had consumed the same illegal street drug, which turned out to have an impurity. The patients were given L-DOPA, which alleviated their motor symptoms. Based on these data, which of these neurotransmission pathways can you infer were NOT disrupted by the impurity?

A. Norepinephrine
B. Serotonin
C. Dopamine
D. Epinephrine
E. All of these have been altered.

Pick the best option.
Then do a quick s ...[truncated]
```

Output (truncated):

```text
FINAL: C



Explanation:

The impurity in the street drug was likely MPTP (1-methyl-4-phenyl-1,2,5,6-tetrahydropyridine), a neurotoxin that selectively destroys dopaminergic neurons. The symptoms described suggest that dopamine transmission was disrupted, but not norepinephrine, serotonin, or epinephrine transmission.

The best answer is C.
```

### T5_temp0.2 — Q8

- Parsed answer: `D` | Correct: `True`

Prompt (truncated):

```text
Free ammonia is captured by synthesis of   (1)  , an amino acid, and then transferred to the amino acid   (2)  . This amino acid (2) provides amino groups for amino acid synthesis.

A. 1=glutamate; 2=glutamine.
B. 1=glutamate; 2=glutamate.
C. 1=glutamine; 2=glutamine.
D. 1=glutamine; 2=glutamate.
E. Ammonia cannot be incorporated into amino acids.

Pick the best option.
Then do a quick self-check: try to find a reason your choice could be wrong.
Finally commit to ONE letter.
Do NOT show the self-check.

You MUST follow this format strictly:
First line: FINAL: <A/B/C/D/E>
After that, you may ad ...[truncated]
```

Output (truncated):

```text
FINAL: D



Explanation: The amino acid glutamine is synthesized from ammonia and α-ketoglutarate. Glutamate can donate an amino group to form α-ketoglutarate, which then reacts with ammonia to synthesize glutamine.



The final answer is D.
```

### T5_temp0.2 — Q9

- Parsed answer: `` | Correct: `False`

Prompt (truncated):

```text
You keep a pet rat, Poppy, in your room. You come home after a long day, and Poppy is hungry – he hasn’t eaten in almost 12 hours. When you open the door to the cabinet where you store the rat food, your cat jumps out lunges toward Poppy. Poppy is scared and has clearly activated the “fight or flight” response, and his blood glucose levels rise rapidly. Which combination of molecular responses has been activated in Poppy’s tissues?

A. ↑ AMPK activity, ↓ mTOR activity, ↓ fructose-2,6-bisphosphate, ↑ glycogen synthase activity
B. ↑ cAMP, ↑ PKA activity, ↓ fructose-2,6-bisphosphate, ↑ gluconeoge ...[truncated]
```

Output (truncated):

```text

```

## How might we improve this experiment?

- Improve answer parsing robustness (some models output 'Answer: B' or extra tokens).

- Add more temperatures and more K to better separate randomness vs systematic prompt effects.

- Add a second model (e.g., different GGUF) to test transferability of the best prompt.

- Add additional metrics: entropy of answers, majority-vote accuracy, calibration-like measures.

## How to expand to large-scale automation?

- Expand dataset: more exam questions or a public MC dataset.

- Add batch runner + caching + parallelization.

- Log hardware/time cost, and normalize cost per correct answer.
