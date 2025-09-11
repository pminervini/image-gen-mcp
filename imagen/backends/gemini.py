"""Gemini image generation backend using the official google-genai SDK.

This backend intentionally keeps the flow minimal and matches Google’s docs:
- Image generation: https://ai.google.dev/gemini-api/docs/image-generation
- Migration guide: https://ai.google.dev/gemini-api/docs/migrate
"""

import base64
import io
import os
from typing import Optional, Tuple

from .base import ImageBackend, ImageResult


def _parse_size(size: str) -> Tuple[int, int]:
    try:
        w_s, h_s = size.lower().split("x", 1)
        w, h = int(w_s), int(h_s)
        w = max(16, min(w, 4096))
        h = max(16, min(h, 4096))
        return w, h
    except Exception:
        return 1024, 1024


class GeminiBackend(ImageBackend):
    name = "gemini"

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        # Per docs, use the image preview model by default.
        # https://ai.google.dev/gemini-api/docs/image-generation
        self.model_name = model_name or "gemini-2.5-flash-image-preview"
        self.api_key = api_key

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        fmt: str = "png",
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> ImageResult:
        api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

        # Desired output format/mime
        fmt_l = fmt.lower()
        desired_mime = f"image/{'jpeg' if fmt_l == 'jpg' else fmt_l}"

        # Compose prompt; keep it simple and human-readable
        width, height = _parse_size(size)
        full_prompt = prompt
        if negative_prompt:
            full_prompt += f"\nNegative prompt: {negative_prompt}"
        # Encourage target size (the model may not guarantee exact dimensions)
        full_prompt += f"\nTarget size: {width}x{height}"

        # Use the new google-genai client
        try:
            import google.genai as genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as e:  # pragma: no cover - import-time environment
            raise RuntimeError(
                "google-genai is required for the Gemini backend. Install with: pip install google-genai"
            ) from e

        client = genai.Client(api_key=api_key)

        # Ask explicitly for IMAGE output; do not set response_mime_type.
        # The server only allows text mime types there.
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            media_resolution=getattr(types.MediaResolution, "MEDIA_RESOLUTION_HIGH", None),
            seed=seed,
        )

        # Generate image from text prompt
        resp = client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=config,
        )

        # Extract first inline image from response
        content_bytes: Optional[bytes] = None
        content_type: Optional[str] = None
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and isinstance(getattr(inline, "data", None), (bytes, str)):
                    data = inline.data
                    if isinstance(data, str):
                        data = base64.b64decode(data)
                    content_bytes = data
                    content_type = getattr(inline, "mime_type", None) or desired_mime
                    break
            if content_bytes:
                break

        if not content_bytes:
            raise RuntimeError("Gemini response did not include any inline image data")

        # Best-effort resize/convert to requested format using Pillow.
        try:
            from PIL import Image  # type: ignore

            img = Image.open(io.BytesIO(content_bytes))
            if img.size != (width, height):
                img = img.resize((width, height))

            buf = io.BytesIO()
            fmt_upper = "JPEG" if fmt_l == "jpg" else fmt_l.upper()
            img.save(buf, format=fmt_upper)
            content_bytes = buf.getvalue()
            content_type = f"image/{'jpeg' if fmt_l == 'jpg' else fmt_l}"
        except Exception:
            # If Pillow fails, keep original bytes and best-guess content_type
            if not content_type:
                content_type = desired_mime

        ext = "jpg" if (content_type or desired_mime).lower().endswith("jpeg") else (content_type or desired_mime).split("/")[-1]
        filename = f"gemini_{abs(hash(prompt)) % 1_000_000}.{ext}"
        return ImageResult(content=content_bytes, content_type=content_type, format=ext, filename=filename)
