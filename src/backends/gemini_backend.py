# src/backends/gemini_backend.py
from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any

from google import genai

from .types import GenerationResult


def _usage_to_dict(usage_obj) -> Dict[str, Any]:
    if usage_obj is None:
        return {}
    d = {}
    for k in ["prompt_token_count", "candidates_token_count", "total_token_count"]:
        if hasattr(usage_obj, k):
            d[k] = getattr(usage_obj, k)
    return d


class GeminiBackend:
    """
    Gemini API backend.
    Keeps the same interface as GPT4AllBackend so the rest of the pipeline doesn't change.
    """

    def __init__(
        self,
        model_name: str = "models/gemini-3-flash-preview",
        rpm_limit: int = 5,
    ):
        self.model_name = model_name
        self.rpm_limit = max(1, rpm_limit)
        self.min_interval = 60.0 / self.rpm_limit

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
        top_p: float = 0.95,            # keep for compatibility, Gemini may ignore
        repeat_penalty: float = 1.1,    # keep for compatibility, Gemini may ignore
        seed: Optional[int] = None,
    ) -> GenerationResult:
        # free-tier rate limit
        now = time.time()
        elapsed = now - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                # Note: Gemini may not support seed or repeat_penalty the same way.
            },
        )
        self._last_call_ts = time.time()

        text = getattr(response, "text", "") or ""
        usage = _usage_to_dict(getattr(response, "usage_metadata", None))

        # Try to map token counts if available
        input_tokens = usage.get("prompt_token_count")
        output_tokens = usage.get("candidates_token_count")  # sometimes present
        # total_tokens = usage.get("total_token_count")  # optional if you want to log

        return GenerationResult(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_count_method="gemini",
        )