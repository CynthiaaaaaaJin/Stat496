# from __future__ import annotations

# import json
# import os
# import time
# from dataclasses import dataclass
# from typing import Optional

# from openai import OpenAI


# @dataclass
# class BackendResponse:
#     text: str
#     token_count_method: str = "openai"
#     input_tokens: int = 0
#     output_tokens: int = 0


# class ChatGPTBackend:
#     """
#     OpenAI (ChatGPT) backend using the Responses API.

#     - Uses Structured Outputs (json_schema) to force JSON format.
#     - Enforces RPM throttling to protect your org limits.
#     """

#     def __init__(
#         self,
#         model_name: str,
#         rpm_limit: int = 500,
#         api_key: Optional[str] = None,
#         base_url: Optional[str] = None,
#     ) -> None:
#         self.model_name = model_name
#         self.rpm_limit = max(1, int(rpm_limit))
#         self._min_interval = 60.0 / float(self.rpm_limit)
#         self._last_call_ts = 0.0

#         # OpenAI SDK client
#         kwargs = {}
#         if api_key:
#             kwargs["api_key"] = api_key
#         if base_url:
#             kwargs["base_url"] = base_url

#         # If api_key is not passed, SDK uses OPENAI_API_KEY env var.
#         self.client = OpenAI(**kwargs)

#     def _throttle(self) -> None:
#         now = time.time()
#         dt = now - self._last_call_ts
#         if dt < self._min_interval:
#             time.sleep(self._min_interval - dt)
#         self._last_call_ts = time.time()

#     def generate(
#         self,
#         prompt: str,
#         temperature: float = 0.2,
#         max_tokens: int = 256,
#         top_p: float = 0.95,
#         repeat_penalty: float = 1.0,
#         seed: Optional[int] = None,
#         allow_explanation: bool = True,
#     ) -> BackendResponse:
#         """
#         Returns:
#           BackendResponse.text -> JSON string (e.g. {"answer":"A","explanation":"..."})
#         """

#         self._throttle()

#         # Structured Outputs JSON schema:
#         # Keep it stable across treatments to avoid confounds.
#         schema = {
#             "type": "object",
#             "additionalProperties": False,
#             "properties": {
#                 "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
#                 "explanation": {"type": "string"},
#             },
#             "required": ["answer", "explanation"],
#         }

#         # If you truly want *no* explanation, you can still keep the key
#         # but instruct the model to leave it empty.
#         explanation_instruction = (
#             "Provide a brief explanation in the explanation field."
#             if allow_explanation
#             else "Set explanation to an empty string."
#         )

#         # IMPORTANT: Put the output contract in the request, not mixed into your treatments.
#         # Your build_prompt() can keep treatments identical; backend enforces JSON schema.
#         system_contract = (
#             "Return ONLY valid JSON matching the provided JSON schema. "
#             "Do not wrap in markdown or backticks. "
#             f"{explanation_instruction}"
#         )

#         # Build input for Responses API
#         input_messages = [
#             {
#                 "role": "system",
#                 "content": [{"type": "input_text", "text": system_contract}],
#             },
#             {
#                 "role": "user",
#                 "content": [{"type": "input_text", "text": prompt}],
#             },
#         ]

#         # Make request
#         # Notes:
#         # - The Responses API supports temperature/top_p/max_output_tokens.
#         # - "seed" may or may not be supported depending on model; we pass only if not None.
#         req = dict(
#             model=self.model_name,
#             input=input_messages,
#             temperature=float(temperature),
#             top_p=float(top_p),
#             max_output_tokens=int(max_tokens),
#             # Structured Outputs for Responses API:
#             text={
#                 "format": {
#                     "type": "json_schema",
#                     "name": "mcq_answer",
#                     "strict": True,
#                     "schema": schema,
#                 }
#             },
#         )
#         if seed is not None:
#             req["seed"] = int(seed)

#         resp = self.client.responses.create(**req)

#         # Extract text output. In the Python SDK, response.output_text aggregates output_text items.
#         # With Structured Outputs, this should be the JSON string.
#         text = getattr(resp, "output_text", "") or ""

#         # Usage tokens (best effort; fields depend on SDK version)
#         input_tokens = 0
#         output_tokens = 0
#         usage = getattr(resp, "usage", None)
#         if usage:
#             input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
#             output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

#         # Safety fallback: sometimes extra whitespace/newlines occur; trim.
#         text = text.strip()

#         return BackendResponse(
#             text=text,
#             token_count_method="openai",
#             input_tokens=input_tokens,
#             output_tokens=output_tokens,
#         )
from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any

from openai import OpenAI
from openai import BadRequestError

from .types import GenerationResult


class ChatGPTBackend:
    """
    OpenAI API backend (plain text).
    IMPORTANT: Does NOT enforce JSON / structured outputs.
    This lets your prompt contract control the output, e.g. last line: FINAL: D
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        rpm_limit: int = 60,
        debug: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.rpm_limit = max(1, rpm_limit)
        self.min_interval = 60.0 / self.rpm_limit
        self.debug = debug

        # API key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY. Set it with: export OPENAI_API_KEY='...'")

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self._last_call_ts = 0.0

    def _throttle(self) -> None:
        now = time.time()
        elapsed = now - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 256,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,  # kept for compatibility; OpenAI ignores
        seed: Optional[int] = None,
    ) -> GenerationResult:
        self._throttle()

        # Responses API input (plain text)
        req: Dict[str, Any] = {
            "model": self.model_name,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "max_output_tokens": int(max_tokens),
        }

        # Many OpenAI models support sampling; some reasoning models reject temperature/top_p.
        # We'll include them first, and if OpenAI returns "unsupported parameter", retry without.
        req_with_sampling = dict(req)
        req_with_sampling["temperature"] = float(temperature)
        req_with_sampling["top_p"] = float(top_p)
        if seed is not None:
            req_with_sampling["seed"] = int(seed)

        t0 = time.time()
        try:
            resp = self.client.responses.create(**req_with_sampling)
        except BadRequestError as e:
            msg = str(e)
            if "Unsupported parameter" in msg and ("temperature" in msg or "top_p" in msg or "seed" in msg):
                # Retry without sampling knobs
                resp = self.client.responses.create(**req)
            else:
                raise
        finally:
            self._last_call_ts = time.time()

        # Extract output text
        text = (getattr(resp, "output_text", "") or "").strip()

        # Usage tokens (best-effort; depends on SDK/version)
        input_tokens = None
        output_tokens = None
        usage = getattr(resp, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_count_method="openai",
        )