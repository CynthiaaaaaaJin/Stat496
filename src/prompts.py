def format_question(q: dict) -> str:
    # supports A-D or A-E if E exists
    letters = [k for k in ["A","B","C","D","E"] if k in q]
    choice_lines = "\n".join([f"{L}. {q[L]}" for L in letters])

    return (
        f"Question (QID={q['qid']}):\n{q['stem']}\n\n"
        f"Choices:\n{choice_lines}\n"
    )

def treatment_prompt(treatment: str, q: dict) -> str:
    base = format_question(q)

    common_rules = (
        "You are answering a multiple-choice question.\n"
        "Return ONLY the final choice letter on the last line exactly as:\n"
        "FINAL: <LETTER>\n"
        "Valid letters are A/B/C/D/E.\n"
    )

    if treatment == "T0":
        return base + "\n" + common_rules + "\nAnswer normally."
    if treatment == "T1":
        return base + "\n" + common_rules + "\nBe concise. Output only the FINAL line."
    if treatment == "T2":
        return base + "\n" + common_rules + "\nReason step-by-step, then output ONLY the FINAL line."
    if treatment == "T3":
        return (
            base + "\n" + common_rules +
            "\nChoose the best option and justify ONLY using the provided choices as evidence.\n"
            "Quote a short phrase from the option you selected.\n"
            "Then output ONLY the FINAL line.\n"
        )
    if treatment == "T4":
        return (
            base + "\n" + common_rules +
            "\nWork step-by-step and cite the provided choices as evidence.\n"
            "Quote a short phrase from the option you selected.\n"
            "Then output ONLY the FINAL line.\n"
        )
    if treatment == "T5":
        return (
            base + "\n" + common_rules +
            "\nAfter selecting an answer, do a self-check: briefly consider one alternative and why it is wrong.\n"
            "Then output ONLY the FINAL line.\n"
        )
    raise ValueError(f"Unknown treatment: {treatment}")
