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
        max_tokens: int = 256,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        now = time.time()
        elapsed = now - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        config: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

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
