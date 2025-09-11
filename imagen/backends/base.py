# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageResult:
    content: bytes
    content_type: str
    format: str
    filename: str


class ImageBackend:
    name: str = "base"

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        fmt: str = "png",
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> ImageResult:
        raise NotImplementedError
