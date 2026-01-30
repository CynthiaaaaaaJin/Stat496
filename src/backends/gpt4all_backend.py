from typing import Dict, Any, Optional
from gpt4all import GPT4All


class GPT4AllBackend:
    """
    Runs a local GGUF model via GPT4All python package.
    You must supply model_path to an existing .gguf file.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        # allow_download=False is important to avoid it trying to fetch online
        self.model = GPT4All(model_name=model_path, model_path="", allow_download=False)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 128,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        # GPT4All doesn't always return token usage in a standard way.
        # We'll store None; you can approximate tokens via len(text)/4 later if needed.
        output = self.model.generate(
            prompt,
            temp=temperature,
            max_tokens=max_output_tokens,
        )
        return {
            "raw_text": output,
            "usage": {},
            "model": self.model_path,
        }
