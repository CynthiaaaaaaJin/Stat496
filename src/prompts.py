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

    letters = [k for k in["A","B","C","D","E"] if k in q]
    valid = "/".join(letters)

    common_rules = (
        "You are answering a multiple-choice question.\n"
        f"Valid answer letters:{valid}.\n"
        "Output rules:\n"
        "- The Last line must be exactly: Final:<LETTER>\n"
        "- Do NOT output anything else after the FINAL line.\n"
    )

    if treatment == "T0":
        return base + "\n" + common_rules + "\nAnswer normally."
    if treatment == "T1":
        return base + "\n" + common_rules + "\nBe concise. Output only the FINAL line."
    if treatment == "T2":
        return (
            base + "\n" + common_rules + 
            "\nBefore the Final line, write 2-4 short steps(each <= 12 words) under:\n"
            "STEPS:\n"
            "Then output ONLY the FINAL line."
        )
    if treatment == "T3":
        return (
            base + "\n" + common_rules +
            "\nChoose the best option and justify ONLY using the text of the selected option as evidence.\n"
            "Before the Final line, output exactly one line:\n"
            "Evidence:\n"
            "Quote a short phrase from the option you selected.\n"
            "Then output ONLY the FINAL line.\n"
        )
    if treatment == "T4":
        return (
            base + "\n" + common_rules +
            "\nWork with 2-4 short steps(each <=12 words).\n"
            "\nChoose the best option and justify ONLY using the text of the selected option as evidence.\n"
            "Before the Final line, output exactly one line:\n"
            "STEPS:\n"
            "(your steps here)\n"
            "Evidence:\n"
            "Quote a short phrase from the option you selected.\n"
            "Then output ONLY the FINAL line.\n"
        )
    if treatment == "T5":
        return (
            base + "\n" + common_rules +
            "\nAfter selecting an answer, do a self-check: briefly consider one alternative and why it is wrong(<=20 words).\n"
            "Before the Final line, output exactly one lines:\n"
            "Self-Check:<ALT_LETTER> - <reason>\n"
            "Then output ONLY the FINAL line.\n"
        )
    raise ValueError(f"Unknown treatment: {treatment}")
