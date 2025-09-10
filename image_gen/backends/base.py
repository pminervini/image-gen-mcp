from __future__ import annotations

from dataclasses import dataclass


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
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> ImageResult:
        raise NotImplementedError

