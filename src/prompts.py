from __future__ import annotations
from typing import Dict, Optional

LETTERS = ["A","B","C","D"] #delete"E"

def format_mcq(stem: str, options: Dict[str, str]) -> str:
    opt_lines = "\n".join([f"{L}. {options.get(L,'')}" for L in LETTERS])
    return f"{stem}\n\n{opt_lines}"

# Strong, parseable contract:
COMMON_RULE = (
    "You MUST follow this format strictly:\n"
    "First line: FINAL: <ANSWER>\n"
    "After that, you may add explanation if allowed.\n"
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
            "Do NOT show the self-check."
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

    # For MCQ, constrain answer space explicitly for easier parsing + fewer invalid outputs
    if is_mcq:
        answer_space = "Your FINAL answer must be exactly one letter: A, B, C, D, or E."
    else:
        answer_space = "Your FINAL answer must be the final result (a short phrase or a number)."

    # If user wants to force no explanation, we still keep FINAL contract.
    explain_rule = "After the first line, add no extra text." if not allow_explanation else "After the first line, you may add explanation."

    return (
        f"{q}\n\n"
        f"{inst}\n"
        f"{answer_space}\n\n"
        f"{COMMON_RULE.replace('After that, you may add explanation if allowed.', explain_rule)}"
    )
