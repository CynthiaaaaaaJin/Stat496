from __future__ import annotations
from typing import Optional

try:
    from gpt4all import GPT4All
except Exception as e:  # pragma: no cover
    GPT4All = None

from .types import GenerationResult

class GPT4AllBackend:
    """Small wrapper so the rest of the code doesn't depend on GPT4All's exact API."""

    def __init__(self, model_filename: str, device: str | None = None):
        if GPT4All is None:
            raise ImportError(
                "gpt4all is not installed or failed to import. "
                "Install with: pip install gpt4all"
            )
        # device is optional; GPT4All will pick a default
        self.model = GPT4All(model_filename, device=device)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 256,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        # GPT4All supports 'prompt', 'temp', 'top_p', etc. The exact names differ across versions.
        # We keep this defensive.
        kwargs = dict(
            prompt=prompt,
            temp=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )
        if seed is not None:
            kwargs["seed"] = seed

        out = self.model.generate(**kwargs)

        # Token counts may not be provided by GPT4All depending on model/tokenizer.
        # We keep them as None if unavailable.
        return GenerationResult(text=str(out), input_tokens=None, output_tokens=None, token_count_method="gpt4all")
