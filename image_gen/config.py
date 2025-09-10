# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass


@dataclass
class Settings:
    host: str = os.getenv("IMAGE_GEN_HOST", "0.0.0.0")
    port: int = int(os.getenv("IMAGE_GEN_PORT", "8080"))
    backend: str = os.getenv("IMAGE_GEN_BACKEND", os.getenv("BACKEND", "auto"))
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")


def get_settings() -> Settings:
    return Settings()
