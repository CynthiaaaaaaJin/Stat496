# Prompt Templates and Temperature Effects on LLM Commonsense Reasoning

**STAT 496 Undergraduate Capstone Project | University of Washington**  
**Authors:** Yan Peng and Yuxin Jin  
**Instructor:** Anne Wagner

## Overview

This project investigates how prompt design and sampling temperature affect the **accuracy and reproducibility** of large language models (LLMs) on commonsense reasoning tasks.

Using 100 questions from the [COSMOS QA](https://wilburone.github.io/cosmos/) dataset, we evaluated three ChatGPT models (GPT-3.5-turbo, GPT-4o-mini, and GPT-4.1-mini) across six prompt treatments and six temperature settings, with five repeated runs per configuration.

## Methods

- **Experimental design:** Six prompt treatments (T0–T5) and six temperature settings (0.2–2.0).
- **Statistical analysis:** Binomial logistic regression, odds ratios, and confidence intervals.
- **Evaluation metrics:** Accuracy, strict stability, and answer entropy.
- **Trade-off analysis:** Pareto frontier analysis of accuracy and reproducibility.

## Main Findings

- Prompt design generally had a greater impact on performance than temperature alone.
- Grounded prompting (T3) achieved one of the strongest balances between accuracy and consistency.
- Self-check prompting (T5) often increased response variability, particularly at higher temperatures.
- GPT-4.1-mini demonstrated the most balanced overall performance among the three models.

## Repository Structure

| Directory | Description |
|---|---|
| `data/` | COSMOS QA dataset and sampled questions |
| `src/` | Experiment execution, data processing, statistical analysis, and visualization |
| `outputs/` | Experimental results, summaries, and plots |
| `writing/` | Project drafts and notes |

### Key Scripts

| File | Description |
|---|---|
| [`run_experiment_chatgpt.py`](src/run_experiment_chatgpt.py) | Runs API-based experiments |
| [`prompts.py`](src/prompts.py) | Defines prompt treatments |
| [`parsing.py`](src/parsing.py) | Extracts final answers |
| [`analyze_results.py`](src/analyze_results.py) | Analyzes experimental results |
| [`robust_cluster_se.py`](src/robust_cluster_se.py) | Performs cluster-robust standard error analysis |
| [`forest_plot.py`](src/forest_plot.py) | Generates forest plots |

## Running the Code

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/CynthiaaaaaaJin/Stat496.git
cd Stat496
python -m pip install -r requirements.txt
```

The main experimental script is `src/run_experiment_chatgpt.py`, with prompt definitions in `src/prompts.py` and the sampled dataset in `data/COSMOS_100.jsonl`.

Analysis scripts are available in `src/`, and saved experimental results are stored in `outputs/`.

Running new experiments requires OpenAI API access and may incur usage costs. The complete experiment and analysis commands have not been independently verified in a clean environment.

## Report

**Prompt Templates and Temperature Effects on LLM Commonsense Reasoning**  
Yan Peng and Yuxin Jin, STAT 496 Capstone Project, 2026.

The full report contains detailed methodology, statistical analyses, results, and discussion.
