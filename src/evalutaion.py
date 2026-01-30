from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


FINAL_RE = re.compile(r"(?im)^\s*FINAL\s*[:\-]\s*([A-E])\s*[\.\)]?\s*$")
EVIDENCE_RE = re.compile(r"(?im)^\s*EVIDENCE\s*[:\-]\s*([A-E])\s*[\.\)]?\s*$")
SELFCHECK_RE = re.compile(r"(?im)^\s*SELFCHECK\s*[:\-]\s*([A-E])\s*(?:[\-–—:]\s*)?(.*)\s*$")
STEPS_HEADER_RE = re.compile(r"(?im)^\s*STEPS\s*:\s*$")


def _extract_last_match(pattern: re.Pattern, text: str) -> Optional[re.Match]:
    matches = list(pattern.finditer(text or ""))
    return matches[-1] if matches else None


def parse_final(text: str, valid_letters: List[str]) -> Optional[str]:
    m = _extract_last_match(FINAL_RE, text)
    if not m:
        return None
    letter = m.group(1).upper()
    return letter if letter in valid_letters else None


def parse_evidence(text: str, valid_letters: List[str]) -> Optional[str]:
    m = _extract_last_match(EVIDENCE_RE, text)
    if not m:
        return None
    letter = m.group(1).upper()
    return letter if letter in valid_letters else None

@dataclass
class SelfCheck:
    alt_letter: str
    reason: str


def parse_selfcheck(text: str, valid_letters: List[str]) -> Optional[SelfCheck]:
    m = _extract_last_match(SELFCHECK_RE, text)
    if not m:
        return None
    alt = (m.group(1) or "").upper()
    reason = (m.group(2) or "").strip()
    if alt not in valid_letters:
        return None
    return SelfCheck(alt_letter=alt, reason=reason)


def has_steps_block(text: str) -> bool:
    """
    Minimal check:
    - contains a 'STEPS:' header line
    - has at least one non-empty line after it before FINAL
    """
    if not text:
        return False
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if STEPS_HEADER_RE.match(line):
            for j in range(i + 1, len(lines)):
                if FINAL_RE.match(lines[j]):
                    break
                if lines[j].strip():
                    return True
            return False
    return False


def is_final_only(text: str) -> bool:
    """
    True iff output is exactly one non-empty line and it matches FINAL format.
    """
    if not text:
        return False
    lines = [ln for ln in (text.strip().splitlines()) if ln.strip()]
    return len(lines) == 1 and bool(FINAL_RE.match(lines[0]))



def strict_stability(finals: List[Optional[str]]) -> float:
    """1 if all K finals are non-null and identical, else 0."""
    if not finals or any(f is None for f in finals):
        return 0.0
    return 1.0 if len(set(finals)) == 1 else 0.0


def agreement_rate(finals: List[Optional[str]]) -> float:
    """Soft stability: fraction of runs that match the modal final answer."""
    vals = [f for f in finals if f is not None]
    if not vals:
        return 0.0
    c = Counter(vals)
    return max(c.values()) / len(vals)


def token_or_proxy_cost(
    prompt: Optional[str],
    text: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    proxy: str = "chars",
) -> int:
    """
    If tokens are available, return input+output tokens.
    Otherwise use a proxy on prompt+output:
      - proxy='chars' => character count
      - proxy='words' => word count
    """
    if input_tokens is not None and output_tokens is not None:
        return int(input_tokens + output_tokens)

    full = (prompt or "") + "\n" + (text or "")
    if proxy == "words":
        return len(re.findall(r"\S+", full))
    return len(full)



@dataclass
class ComplianceResult:
    final_present: bool
    final_only: Optional[bool] = None

    steps_present: Optional[bool] = None

    evidence_present: Optional[bool] = None
    evidence_matches_final: Optional[bool] = None

    selfcheck_present: Optional[bool] = None
    selfcheck_alt_valid: Optional[bool] = None
    selfcheck_alt_diff_from_final: Optional[bool] = None


def score_compliance(
    treatment: str,
    raw_text: str,
    valid_letters: List[str],
) -> Tuple[Optional[str], ComplianceResult]:
    """
    Returns (parsed_final, compliance_flags).
    """
    final = parse_final(raw_text, valid_letters)
    evidence = parse_evidence(raw_text, valid_letters)
    sc = parse_selfcheck(raw_text, valid_letters)

    res = ComplianceResult(final_present=final is not None)

    if treatment == "T1":
        res.final_only = is_final_only(raw_text)

    if treatment == "T2":
        res.steps_present = has_steps_block(raw_text)

    if treatment in ("T3", "T4"):
        res.evidence_present = evidence is not None
        res.evidence_matches_final = (evidence is not None and final is not None and evidence == final)
        if treatment == "T4":
            res.steps_present = has_steps_block(raw_text)

    if treatment == "T5":
        res.selfcheck_present = sc is not None
        res.selfcheck_alt_valid = sc is not None  # parse_selfcheck enforces valid letter
        res.selfcheck_alt_diff_from_final = (sc is not None and final is not None and sc.alt_letter != final)

    return final, res



def per_question_summary(
    runs: List[Dict[str, Any]],
    valid_letters_map: Dict[str, List[str]],
    cost_proxy: str = "chars",
) -> List[Dict[str, Any]]:
    """
    Summarize K runs for each (qid, treatment, temperature).

    Each element in `runs` should include at least:
      - qid: str/int
      - treatment: str  (T0..T5)
      - temperature: float
      - run: int
      - ground_truth: str (A..E)
      - raw_text: str

    Optional fields:
      - prompt: str
      - input_tokens: int
      - output_tokens: int

    `valid_letters_map[qid]` should be the valid options for that question,
    e.g. ["A","B","C","D"] or ["A","B","C","D","E"].
    """
    # group by (qid, treatment, temperature)
    buckets: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for r in runs:
        key = (str(r["qid"]), str(r["treatment"]), float(r["temperature"]))
        buckets.setdefault(key, []).append(r)

    def mean_bool(vals: List[Optional[bool]]) -> Optional[float]:
        v = [x for x in vals if x is not None]
        if not v:
            return None
        return sum(1 for x in v if x) / len(v)

    out: List[Dict[str, Any]] = []

    for (qid, treatment, temp), lst in buckets.items():
        valid_letters = valid_letters_map.get(qid, ["A", "B", "C", "D"])

        finals: List[Optional[str]] = []
        corrects: List[bool] = []
        costs: List[int] = []
        comps: List[ComplianceResult] = []

        # sort by run index if present
        lst_sorted = sorted(lst, key=lambda x: x.get("run", 0))

        for r in lst_sorted:
            raw = r.get("raw_text", "") or ""
            gt = (r.get("ground_truth", "") or "").strip().upper()

            final, comp = score_compliance(treatment, raw, valid_letters)
            finals.append(final)
            comps.append(comp)

            corrects.append(final is not None and gt in valid_letters and final == gt)

            costs.append(
                token_or_proxy_cost(
                    prompt=r.get("prompt"),
                    text=raw,
                    input_tokens=r.get("input_tokens"),
                    output_tokens=r.get("output_tokens"),
                    proxy=cost_proxy,
                )
            )

        out.append({
            "qid": qid,
            "treatment": treatment,
            "temperature": temp,
            "K": len(lst_sorted),

            # primary outcomes
            "accuracy_over_k": sum(1 for x in corrects if x) / len(corrects) if corrects else 0.0,
            "final_parse_rate": sum(1 for f in finals if f is not None) / len(finals) if finals else 0.0,

            # stability
            "strict_stability": strict_stability(finals),
            "agreement_rate": agreement_rate(finals),

            # cost proxy
            "avg_cost": sum(costs) / len(costs) if costs else 0.0,

            # compliance outcomes (treatment-specific)
            "final_only_rate": mean_bool([c.final_only for c in comps]),
            "steps_present_rate": mean_bool([c.steps_present for c in comps]),
            "evidence_present_rate": mean_bool([c.evidence_present for c in comps]),
            "evidence_match_rate": mean_bool([c.evidence_matches_final for c in comps]),
            "selfcheck_present_rate": mean_bool([c.selfcheck_present for c in comps]),
            "selfcheck_alt_diff_rate": mean_bool([c.selfcheck_alt_diff_from_final for c in comps]),
        })

    return out