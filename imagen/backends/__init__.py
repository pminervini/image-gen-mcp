# -*- coding: utf-8 -*-

from .base import ImageBackend, ImageResult
from ..config import get_settings
import importlib

from typing import Optional


def get_backend(preferred: Optional[str] = None) -> ImageBackend:
    settings = get_settings()
    choice = (preferred or settings.backend or "auto").lower()
    if choice == "mock":
        mod = importlib.import_module("imagen.backends.mock")
        return mod.MockBackend()
    if choice in ("gemini", "google", "imagen"):
        mod = importlib.import_module("imagen.backends.gemini")
        return mod.GeminiBackend(api_key=settings.gemini_api_key)
    if choice in ("qwen", "qwen-image", "qwen_image"):
        mod = importlib.import_module("imagen.backends.qwen")
        return mod.QwenImageBackend()
    if choice in ("hunyuan", "hunyuanimage", "hunyuan-image"):
        mod = importlib.import_module("imagen.backends.hunyuan")
        return mod.HunyuanBackend()
    # auto
    if settings.gemini_api_key:
        mod = importlib.import_module("imagen.backends.gemini")
        return mod.GeminiBackend(api_key=settings.gemini_api_key)
    mod = importlib.import_module("imagen.backends.mock")
    return mod.MockBackend()

__all__ = [
    "ImageBackend",
    "ImageResult",
    # Concrete backend classes are imported lazily
    "get_backend",
]
