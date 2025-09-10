from __future__ import annotations

import io
import os
import random
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from .base import ImageBackend, ImageResult


class MockBackend(ImageBackend):
    name = "mock"

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        fmt: str = "png",
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> ImageResult:
        w, h = _parse_size(size)
        rng = random.Random(seed)
        bg_color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))

        img = Image.new("RGB", (w, h), bg_color)
        draw = ImageDraw.Draw(img)

        # Try to load a default font
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        lines = [
            "Mock Backend",
            prompt[:60] + ("..." if len(prompt) > 60 else ""),
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
        ]

        y = 10
        for line in lines:
            draw.text((10, y), line, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
            y += 20

        out = io.BytesIO()
        fmt_upper = fmt.upper()
        if fmt_upper == "JPG":
            fmt_upper = "JPEG"
        img.save(out, format=fmt_upper)
        content = out.getvalue()
        ext = fmt.lower() if fmt.lower() != "jpeg" else "jpg"
        filename = f"mock_{abs(hash(prompt)) % 1_000_000}.{ext}"
        content_type = f"image/{'jpeg' if fmt_lower(fmt)=='jpg' else fmt.lower()}"
        return ImageResult(content=content, content_type=content_type, format=fmt.lower(), filename=filename)


def _parse_size(size: str) -> tuple[int, int]:
    try:
        w_s, h_s = size.lower().split("x", 1)
        w, h = int(w_s), int(h_s)
        w = max(16, min(w, 4096))
        h = max(16, min(h, 4096))
        return w, h
    except Exception:
        return 1024, 1024


def fmt_lower(fmt: str) -> str:
    f = fmt.lower()
    return "jpg" if f == "jpeg" else f

