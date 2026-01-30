from __future__ import annotations
import re
from typing import Dict, Any, Tuple, Optional

def parse_final_line(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"^\s*FINAL\s*:\s*(.+?)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ""

def parse_mcq_letter(text: str) -> str:
    if not text:
        return ""
    # 1) Strong signal: FINAL: X
    m = re.search(r"FINAL\s*:\s*([A-E])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2) Look at last non-empty line (standalone letter or (A))
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        m2 = re.fullmatch(r"[\(\[]?\s*([A-E])\s*[\)\]]?", ln, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).upper()

    # 3) Fallback: last occurrence of a letter token
    m3 = re.findall(r"\b([A-E])\b", text, flags=re.IGNORECASE)
    return m3[-1].upper() if m3 else ""

def parse_number(text: str) -> str:
    """Heuristic numeric parser: use FINAL line if present; else last number in text."""
    if not text:
        return ""
    final = parse_final_line(text)
    candidate = final if final else text
    # allow integers/decimals/negatives
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", candidate)
    if nums:
        return nums[-1]
    # fallback: find anywhere
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return nums[-1] if nums else ""

def parse_answer(item: Dict[str, Any], model_text: str) -> str:
    t = item.get("type", "mcq")
    fmt = item.get("answer_format", "")
    if t == "mcq" or fmt == "letters":
        return parse_mcq_letter(model_text)
    if fmt == "number":
        return parse_number(model_text)
    # generic text: use FINAL line if available, else last non-empty line
    final = parse_final_line(model_text)
    if final:
        return final
    lines = [ln.strip() for ln in model_text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def is_correct(parsed: str, gt_list: list[str], answer_format: str) -> bool:
    if not parsed or not gt_list:
        return False
    if answer_format == "letters":
        return parsed.strip().upper() in [x.strip().upper() for x in gt_list]
    if answer_format == "number":
        # numeric compare with tolerance for formatting differences
        try:
            p = float(parsed)
            for x in gt_list:
                try:
                    g = float(x)
                    if abs(p - g) <= 1e-6:
                        return True
                except:
                    continue
            return False
        except:
            return False
    # text exact (can be expanded later)
    return parsed.strip() in [x.strip() for x in gt_list]
