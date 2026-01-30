from __future__ import annotations
import json
from typing import Dict, List, Any, Iterable

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def normalize_answer_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip() != ""]
    return [str(v).strip()]

def iter_dataset_items(path: str) -> Iterable[Dict[str, Any]]:
    """Dataset must be JSONL; each line is an item."""
    for item in load_jsonl(path):
        # minimum fields
        if "id" not in item:
            raise ValueError("Each dataset row must have an 'id' field.")
        # defaults
        item.setdefault("type", "mcq" if "options" in item else "freeform")
        item.setdefault("stem", item.get("question", ""))  # allow legacy field
        if "answer" in item:
            item["answer"] = normalize_answer_list(item["answer"])
        else:
            item["answer"] = []
        return_item = {
            "id": str(item["id"]),
            "type": item["type"],
            "stem": str(item.get("stem", "")),
            "options": item.get("options", None),
            "answer": item["answer"],
            "answer_format": item.get("answer_format", "letters" if item.get("options") else "text"),
            "meta": item.get("meta", {}),
            "source": item.get("source", {}),
        }
        yield return_item
