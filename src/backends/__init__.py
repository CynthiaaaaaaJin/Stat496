# src/backends/__init__.py

# Always available (local)
from .gpt4all_backend import GPT4AllBackend

# Optional (only if installed)
try:
    from .gemini_backend import GeminiBackend
except ModuleNotFoundError:
    GeminiBackend = None  # type: ignore
