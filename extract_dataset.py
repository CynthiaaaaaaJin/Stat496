"""
Extract a subset of questions (6,8,9,10,11,12,13,14,15,17) from:
- 406_exam 1_A.pdf
and extract highlighted answers from:
- 406_exam 1_A KEY_UPDATED.pdf

Outputs:
- JSON array + JSONL
- One JSON per question (q06.json, q08.json, ...)
"""

import os, re, json
import pdfplumber
import fitz  # PyMuPDF

TARGET_QS = [6, 8, 14, 15, 17]

def extract_text_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n\n".join(texts)

def extract_block(text: str, qnum: int) -> str:
    m = re.search(rf"(?m)^\s*{qnum}\.\s", text)
    if not m:
        raise ValueError(f"Could not find question {qnum}")
    rest = text[m.end():]
    mnext = re.search(r"(?m)^\s*(\d+)\.\s", rest)
    end = len(text) if not mnext else m.end() + mnext.start()
    return text[m.start():end].strip()

def parse_mcq(block: str):
    lines = [l.rstrip() for l in block.splitlines() if l.strip()]
    first = lines[0]
    qmatch = re.match(r"^\s*(\d+)\.\s*(.*)", first)
    qnum = int(qmatch.group(1))
    after = qmatch.group(2)

    qtext_parts = [after]
    choices = {}
    current = None
    for line in lines[1:]:
        opt = re.match(r"^\s*([A-E])\.\s*(.*)", line)
        if opt:
            current = opt.group(1)
            choices[current] = opt.group(2).strip()
        else:
            if current is None:
                qtext_parts.append(line.strip())
            else:
                choices[current] += " " + line.strip()

    question = re.sub(r"\s+", " ", " ".join(qtext_parts)).strip()
    for k in list(choices.keys()):
        choices[k] = re.sub(r"\s+", " ", choices[k]).strip()
    return qnum, question, choices

def get_question_positions(page):
    words = page.get_text("words")
    positions = []
    for w in words:
        token = w[4]
        m = re.match(r"^(\d+)\.$", token)
        if m:
            positions.append((int(m.group(1)), w[1]))  # y0
    positions.sort(key=lambda x: x[1])
    return positions

def extract_highlight_annots(page):
    annots = []
    a = page.first_annot
    while a:
        if a.type[0] == 8:  # highlight
            quads = a.vertices  # list of (x,y)
            rects = []
            for i in range(0, len(quads), 4):
                pts = quads[i:i+4]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                rects.append(fitz.Rect(min(xs), min(ys), max(xs), max(ys)))
            # union bounds (for y)
            bound = rects[0]
            for r in rects[1:]:
                bound |= r
            annots.append((bound, rects))
        a = a.next
    return annots

def words_in_rects_threshold(page, rects, thresh=0.5):
    words = page.get_text("words")
    selected = []
    for w in words:
        wrect = fitz.Rect(w[0], w[1], w[2], w[3])
        area = wrect.get_area() or 0.0
        if area == 0:
            continue
        best = 0.0
        for r in rects:
            inter = wrect & r
            if inter:
                best = max(best, inter.get_area() / area)
        if best >= thresh:
            selected.append(w)
    selected.sort(key=lambda t: (round(t[1], 1), t[0]))
    return [w[4] for w in selected]

def map_highlights_to_questions(key_pdf: str, target_qs=None, thresh=0.5):
    doc = fitz.open(key_pdf)
    mapping = {}
    for pi in range(doc.page_count):
        page = doc.load_page(pi)
        qpos = get_question_positions(page)
        if not qpos:
            continue
        for bound, rects in extract_highlight_annots(page):
            y = bound.y0
            qnum = None
            for q, qy in qpos:
                if qy <= y + 1e-3:
                    qnum = q
                else:
                    break
            if qnum is None:
                continue
            if target_qs and qnum not in target_qs:
                continue
            txt = " ".join(words_in_rects_threshold(page, rects, thresh=thresh))
            mapping.setdefault(qnum, []).append(txt)
    return mapping

def infer_answer_letters(highlight_texts):
    combined = " ".join(highlight_texts)
    letters = re.findall(r"\b([A-E])\.", combined)
    # unique preserve order
    seen = set()
    out = []
    for l in letters:
        if l not in seen:
            out.append(l)
            seen.add(l)
    return out

def main():
    # If your PDFs are in a different folder, update these two lines:
    exam_pdf = "data/406_exam 1_A.pdf"
    key_pdf  = "data/406_exam 1_A KEY_UPDATED.pdf"

    out_dir  = "outputs/bioc406_exam1A_subset"
    os.makedirs(out_dir, exist_ok=True)

    exam_text = extract_text_pdf(exam_pdf)
    hl = map_highlights_to_questions(key_pdf, target_qs=TARGET_QS, thresh=0.5)

    records = []
    for q in TARGET_QS:
        block = extract_block(exam_text, q)
        qnum, question, choices = parse_mcq(block)
        answers = infer_answer_letters(hl.get(qnum, []))
        rec = {
            "id": f"BIOC406_Exam1A_Q{qnum:02d}",
            "question": question,
            "choices": choices,
            "answer": answers,  # list (Q13 may have multiple highlighted options in key PDF)
            "answer_format": "letters",
            "source": {
                "exam_pdf": exam_pdf,
                "key_pdf": key_pdf,
                "question_number": qnum,
            },
        }
        records.append(rec)

        with open(os.path.join(out_dir, f"q{qnum:02d}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "dataset.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} items to {out_dir}/dataset.json and {out_dir}/dataset.jsonl")

if __name__ == "__main__":
    main()
