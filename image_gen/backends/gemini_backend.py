from __future__ import annotations

import base64
from typing import Optional

from .base import ImageBackend, ImageResult


class GeminiBackend(ImageBackend):
    name = "gemini"

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name or "imagegeneration"

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        fmt: str = "png",
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> ImageResult:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set; cannot use Gemini backend. Set BACKEND=mock to use mock.")

        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:  # pragma: no cover - only hit if dependency missing
            raise RuntimeError("google-generativeai package not available") from e

        genai.configure(api_key=self.api_key)

        # The python SDK supports image generation using the special imagegeneration model.
        # See: https://ai.google.dev/gemini-api/docs/image-generation
        # The exact return types may vary between SDK versions, so we attempt robust extraction.
        model = genai.GenerativeModel(self.model_name)

        kwargs = {}
        # Some SDK versions accept image_size like "1024x1024" or a tuple
        kwargs["size"] = size
        if seed is not None:
            kwargs["seed"] = seed
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        # Attempt to call generate_images; if not available, surface a helpful error.
        if not hasattr(model, "generate_images"):
            raise RuntimeError(
                "The installed google-generativeai SDK does not support image generation via generate_images()."
            )

        result = model.generate_images(prompt=prompt, **kwargs)

        # Try to extract bytes from common representations
        content: bytes | None = None
        if hasattr(result, "images") and result.images:
            first = result.images[0]
            # Common fields: .bytes, .image.image_bytes, .image.as_png()
            if hasattr(first, "bytes") and isinstance(first.bytes, (bytes, bytearray)):
                content = bytes(first.bytes)
            elif hasattr(first, "image") and hasattr(first.image, "bytes"):
                content = bytes(first.image.bytes)  # type: ignore[attr-defined]
            elif hasattr(first, "image") and hasattr(first.image, "as_png"):
                content = first.image.as_png()  # type: ignore[attr-defined]
            elif hasattr(first, "image") and hasattr(first.image, "data"):
                data = first.image.data  # type: ignore[attr-defined]
                if isinstance(data, (bytes, bytearray)):
                    content = bytes(data)
                else:
                    # Sometimes it may be base64-encoded
                    try:
                        content = base64.b64decode(data)
                    except Exception:
                        pass

        if not content:
            # Try direct content field from result
            if hasattr(result, "image") and hasattr(result.image, "as_png"):
                content = result.image.as_png()

        if not content:
            raise RuntimeError("Unable to extract image bytes from Gemini response")

        fmt_lower = fmt.lower()
        content_type = f"image/{'jpeg' if fmt_lower == 'jpg' else fmt_lower}"
        filename = f"gemini_{abs(hash(prompt)) % 1_000_000}.{fmt_lower}"
        return ImageResult(content=content, content_type=content_type, format=fmt_lower, filename=filename)

