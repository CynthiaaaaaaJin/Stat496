import os
import time
from typing import Dict, Any, Optional

from google import genai


def _usage_to_dict(usage_obj) -> Dict[str, Any]:
    """Gemini SDK returns a UsageMetadata object; convert to plain dict safely."""
    if usage_obj is None:
        return {}
    d = {}
    for k in ["prompt_token_count", "candidates_token_count", "total_token_count"]:
        if hasattr(usage_obj, k):
            d[k] = getattr(usage_obj, k)
    return d


class GeminiBackend:
    """
    Uses Google AI Studio / Gemini API key (free tier supported).
    Free tier has strict rate limits, so we sleep between calls.
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
            raise RuntimeError("Missing GEMINI_API_KEY. Set it with: export GEMINI_API_KEY='...'" )

        # Client reads api key from env or you can pass explicitly
        self.client = genai.Client(api_key=api_key)
        self._last_call_ts = 0.0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 128,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        # rate limit (free tier)
        now = time.time()
        elapsed = now - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
        self._last_call_ts = time.time()

        text = getattr(response, "text", "") or ""
        usage = _usage_to_dict(getattr(response, "usage_metadata", None))

        return {
            "raw_text": text,
            "usage": usage,
            "model": self.model_name,
        }
