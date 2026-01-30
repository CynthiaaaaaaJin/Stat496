import re
import json
from pathlib import Path

import pdfplumber

# --------- helpers ---------
def clean(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def parse_questions_from_text(text: str):
    """
    Parse questions like:
      1. ... (A) ... (B) ... (C) ... (D) ...
    Return list of dict: {qid, stem, A,B,C,D,(E)}
    """
    # Split into question blocks starting with "1." at line start-ish
    # Works best after we normalize newlines.
    text = "\n" + text
    q_blocks = re.split(r"\n\s*(\d{1,3})\.\s", text)
    # re.split returns: [prefix, qid1, block1, qid2, block2, ...]
    out = []
    i = 1
    while i < len(q_blocks) - 1:
        qid = int(q_blocks[i])
        block = q_blocks[i + 1]
        i += 2

        block = clean(block)

        # Find choices (A) ... (B) ... etc
        # We'll capture everything between markers.
        # Some PDFs may have line breaks inside choices; this handles that.
        choice_pat = r"\((A|B|C|D|E)\)\s"
        parts = re.split(choice_pat, block)

        # If no choices found, skip (or keep stem only)
        if len(parts) < 3:
            out.append({"qid": qid, "stem": block})
            continue

        stem = clean(parts[0])
        choices = {}
        # parts = [stem, 'A', textA, 'B', textB, ...]
        j = 1
        while j < len(parts) - 1:
            letter = parts[j]
            choice_text = parts[j + 1]
            choices[letter] = clean(choice_text)
            j += 2

        row = {"qid": qid, "stem": stem}
        row.update(choices)
        out.append(row)

    return out

def parse_answers_from_text(text: str):
    """
    Parse answer key like:
      1. B
      2. C
    Returns dict {qid: 'A'/'B'/'C'/'D'/'E'}
    """
    text = clean(text)
    ans = {}
    # common patterns: "1. B" or "1 B"
    for m in re.finditer(r"\b(\d{1,3})\s*[\.\)]?\s*([A-E])\b", text):
        qid = int(m.group(1))
        letter = m.group(2)
        # keep first occurrence
        if qid not in ans:
            ans[qid] = letter
    return ans

def extract_pdf_text(pdf_path: Path) -> str:
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                all_text.append(t)
    return "\n".join(all_text)

# --------- main ---------
def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    q_pdf = data_dir / "AP Chemistry 60 Multiple Choices.pdf"
    a_pdf = data_dir / "AP Chemistry 60 Multiple Choices Answer.pdf"

    out_jsonl = data_dir / "apchem_60.jsonl"
    out_ans_json = data_dir / "apchem_60_answers.json"

    if not q_pdf.exists():
        raise FileNotFoundError(f"Missing question PDF: {q_pdf}")
    if not a_pdf.exists():
        raise FileNotFoundError(f"Missing answer PDF: {a_pdf}")

    print("Extracting question PDF text...")
    q_text = extract_pdf_text(q_pdf)
    questions = parse_questions_from_text(q_text)
    print(f"Parsed questions: {len(questions)}")

    print("Extracting answer PDF text...")
    a_text = extract_pdf_text(a_pdf)
    answers = parse_answers_from_text(a_text)
    print(f"Parsed answers: {len(answers)}")

    # merge
    merged = []
    missing_choice = 0
    missing_answer = 0

    for q in questions:
        row = dict(q)
        # enforce choices exist for your prompts
        letters = [L for L in ["A", "B", "C", "D", "E"] if L in row]
        if not letters:
            missing_choice += 1

        if q["qid"] in answers:
            row["gold"] = answers[q["qid"]]
        else:
            row["gold"] = None
            missing_answer += 1

        merged.append(row)

    # write outputs
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with out_ans_json.open("w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_ans_json}")
    print(f"Missing choices rows: {missing_choice}")
    print(f"Missing answers rows: {missing_answer}")
    print("Done.")

if __name__ == "__main__":
    main()
