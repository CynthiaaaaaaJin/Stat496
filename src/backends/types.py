from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerationResult:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    token_count_method: str = "unknown"
