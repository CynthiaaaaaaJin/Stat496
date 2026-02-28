from __future__ import annotations
from typing import Dict, Optional

LETTERS = ["A", "B", "C", "D"]

def format_mcq(stem: str, options: Dict[str, str]) -> str:
    opt_lines = "\n".join([f"{L}. {options.get(L, '')}" for L in LETTERS])
    return f"{stem}\n\n{opt_lines}"

# Strong, parseable contract:
COMMON_RULE = (
    "You MUST follow this format strictly:\n"
    "Last line: FINAL: <ANSWER>\n"
    "Explain before the last line.\n"
    "Do NOT change the word FINAL.\n"
)

def _treatment_instruction(treatment: str, is_mcq: bool) -> str:
    # Keep treatments consistent across tasks; adapt wording for non-MCQ.
    if treatment == "T0":
        return "Answer the question."
    if treatment == "T1":
        return "Answer with ONLY the final result (no extra explanation)."
    if treatment == "T2":
        return "Answer and include step-by-step reasoning."
    if treatment == "T3":
        if is_mcq:
            return "Use only the provided options/background; do not use outside knowledge."
        return "Use only the information in the question; avoid unstated assumptions or outside facts."
    if treatment == "T4":
        if is_mcq:
            return "Use step-by-step reasoning AND only cite/use the provided options/background."
        return "Use step-by-step reasoning AND avoid unstated assumptions; rely only on the question text."
    if treatment == "T5":
        return (
            "Pick the best answer.\n"
            "Then do a quick self-check: try to find a reason your choice could be wrong.\n"
            "Finally commit to ONE final answer.\n"
        )
    raise ValueError(f"Unknown treatment: {treatment}")

def build_prompt(
    treatment: str,
    stem: str,
    options: Optional[Dict[str, str]] = None,
    allow_explanation: bool = True,
) -> str:
    is_mcq = options is not None
    q = format_mcq(stem, options) if is_mcq else stem

    inst = _treatment_instruction(treatment, is_mcq)

    # Force T1 to be final-only regardless of caller setting
    if treatment == "T1":
        allow_explanation = False

    # Constrain answer space explicitly (MCQ vs non-MCQ)
    if is_mcq:
        answer_space = "Your FINAL answer must be exactly one letter: A, B, C, D."
    else:
        answer_space = "Your FINAL answer must be the final result (a short phrase or a number)."

    # Explanation rule toggled by allow_explanation
    explain_rule = (
        "Before the last line, add no extra text.\n"
        if not allow_explanation
        else "Explain before the last line.\n"
    )

    # Replace the explain line inside COMMON_RULE to avoid conflicts/duplication
    rules = COMMON_RULE.replace("Explain before the last line.\n", explain_rule)

    # Put requirements BEFORE the question text
    return (
        f"{inst}\n"
        f"{answer_space}\n"
        f"{rules}\n\n"
        f"QUESTION:\n{q}\n"
    )