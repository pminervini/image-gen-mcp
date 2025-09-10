from .base import ImageBackend, ImageResult
from .mock_backend import MockBackend
from .gemini_backend import GeminiBackend
from ..config import get_settings
import importlib

def get_backend(preferred: str | None = None) -> ImageBackend:
    settings = get_settings()
    choice = (preferred or settings.backend or "auto").lower()
    if choice == "mock":
        return MockBackend()
    if choice in ("gemini", "google", "imagen"):
        return GeminiBackend(api_key=settings.gemini_api_key)
    if choice in ("qwen", "qwen-image", "qwen_image"):
        mod = importlib.import_module("image_gen.backends.qwen_image_backend")
        return mod.QwenImageBackend()
    if choice in ("hunyuan", "hunyuanimage", "hunyuan-image"):
        mod = importlib.import_module("image_gen.backends.hunyuan_backend")
        return mod.HunyuanBackend()
    # auto
    if settings.gemini_api_key:
        return GeminiBackend(api_key=settings.gemini_api_key)
    return MockBackend()

__all__ = [
    "ImageBackend",
    "ImageResult",
    "MockBackend",
    "GeminiBackend",
    "get_backend",
]
