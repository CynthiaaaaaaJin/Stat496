# src/backends/gemini_backend.py
from __future__ import annotations

import os
import sys
import time
from typing import Optional, Dict, Any

from google import genai

from .types import GenerationResult


def _usage_to_dict(usage_obj) -> Dict[str, Any]:
    if usage_obj is None:
        return {}
    d: Dict[str, Any] = {}
    for k in ["prompt_token_count", "candidates_token_count", "total_token_count"]:
        if hasattr(usage_obj, k):
            d[k] = getattr(usage_obj, k)
    return d


def _extract_text(response) -> str:
    """
    Gemini SDK sometimes returns empty response.text even when there is text in candidates.
    This function tries response.text first, then falls back to candidates->content->parts.
    """
    t = getattr(response, "text", None)
    if t:
        return str(t)

    cands = getattr(response, "candidates", None) or []
    for c in cands:
        content = getattr(c, "content", None)
        parts = getattr(content, "parts", None) or []
        for p in parts:
            pt = getattr(p, "text", None)
            if pt:
                return str(pt)

    return ""


class GeminiBackend:
    """
    Gemini API backend.
    Keeps the same interface as GPT4AllBackend so the rest of the pipeline doesn't change.
    """

    def __init__(
        self,
        model_name: str = "models/gemini-3-flash-preview",
        rpm_limit: int = 5,
        debug: bool = False,
    ):
        self.model_name = model_name
        self.rpm_limit = max(1, rpm_limit)
        self.min_interval = 60.0 / self.rpm_limit
        self.debug = debug

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY. Set it with: export GEMINI_API_KEY='...'")

        self.client = genai.Client(api_key=api_key)
        self._last_call_ts = 0.0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 256,          # unify naming with GPT4AllBackend
        top_p: float = 0.95,            # kept for compatibility; may be ignored
        repeat_penalty: float = 1.1,    # kept for compatibility; may be ignored
        seed: Optional[int] = None,     # Gemini may ignore
    ) -> GenerationResult:
        # free-tier rate limit (basic)
        now = time.time()
        elapsed = now - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        # Build config. Only include fields that are commonly accepted.
        config: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Optional: if your SDK/model supports top_p, uncomment this:
        # config["top_p"] = top_p

        # Optional: stabilize plain text output if supported (safe to try; remove if errors):
        # config["response_mime_type"] = "text/plain"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        self._last_call_ts = time.time()

        text = _extract_text(response)
        usage = _usage_to_dict(getattr(response, "usage_metadata", None))

        input_tokens = usage.get("prompt_token_count")
        output_tokens = usage.get("candidates_token_count")

        if self.debug and not text:
            try:
                sys.stderr.write("WARN: Gemini returned empty text. Dumping response (truncated):\n")
                # model_dump_json exists on pydantic models; if unavailable, fallback to str()
                if hasattr(response, "model_dump_json"):
                    sys.stderr.write(response.model_dump_json()[:2000] + "\n")
                else:
                    sys.stderr.write(str(response)[:2000] + "\n")
            except Exception:
                pass

        return GenerationResult(
            text=(text or "").strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_count_method="gemini",
        )