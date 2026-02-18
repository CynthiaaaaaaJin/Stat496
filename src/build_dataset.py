# src/build_dataset.py
import json
from pathlib import Path
from typing import Any, Dict, List, Union


def load_json(path: Union[str, Path]) -> Any:
    """Load a standard JSON file."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def zfill_qid(qid: Union[str, int], width: int = 2) -> str:
    """Convert qid like '6' -> '06' (for EXAM_Q06)."""
    return str(qid).strip().lstrip("Qq").zfill(width)


def build_dataset(
    questions: List[Dict[str, Any]],
    answers_map: Dict[str, List[str]],
    exam_prefix: str = "EXAM",
    exam_pdf: str = "data/406_exam_1_A.pdf",
    key_pdf: str = "data/406_exam 1_A KEY_UPDATED.pdf",
    include_answer_text: bool = True,
) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []

    for q in questions:
        if "qid" not in q:
            raise KeyError(f"Question missing 'qid': {q}")

        qnum_raw = str(q["qid"]).strip()
        qnum_int = int(qnum_raw)
        qnum_pad = zfill_qid(qnum_raw, width=2)

        # answers_clean.json might use "6" keys, not "06"
        ans_letters = answers_map.get(qnum_raw)
        if ans_letters is None:
            # fallback if answers are padded
            ans_letters = answers_map.get(qnum_pad, [])
        if ans_letters is None:
            ans_letters = []

        item = {
            "id": f"{exam_prefix}_Q{qnum_pad}",
            "type": "mcq",
            "stem": q.get("stem", ""),
            "options": q.get("options", {}),
            "answer": ans_letters,
            "answer_format": "letters",
            "source": {
                "exam_pdf": exam_pdf,
                "key_pdf": key_pdf,
                "question_number": qnum_int,
            },
        }

        # Add "words" for the answer: map letter -> option text
        if include_answer_text:
            opts = item["options"] or {}
            item["answer_text"] = [opts.get(letter, "") for letter in ans_letters]

        dataset.append(item)

    return dataset


def save_jsonl(items: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_json(items: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    # Project root = parent of src/
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    questions_path = data_dir / "questions_selected.json"
    answers_path = data_dir / "answers_clean.json"

    # ✅ renamed outputs
    out_jsonl = data_dir / "sat_subset.jsonl"
    out_json = data_dir / "sat_subset.json"

    questions = load_json(questions_path)
    answers_map = load_json(answers_path)

    if not isinstance(questions, list):
        raise TypeError(f"{questions_path} should be a JSON list of question objects.")
    if not isinstance(answers_map, dict):
        raise TypeError(f"{answers_path} should be a JSON object mapping qid -> answers.")

    dataset = build_dataset(
        questions=questions,
        answers_map=answers_map,
        exam_prefix="EXAM",
        exam_pdf="data/406_exam_1_A.pdf",
        key_pdf="data/406_exam 1_A KEY_UPDATED.pdf",
        include_answer_text=True,
    )

    save_jsonl(dataset, out_jsonl)
    save_json(dataset, out_json)

    print(f"✅ Built {len(dataset)} items")
    print(f"✅ Wrote: {out_jsonl}")
    print(f"✅ Wrote: {out_json}")


if __name__ == "__main__":
    main()
