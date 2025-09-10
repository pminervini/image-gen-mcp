#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Usage examples (direct CLI, no MCP):
#   image-gen "A red square" --backend mock --fmt png --output red.png
#   image-gen "A cat photo" --backend gemini --fmt jpg --output cat.jpg
#   image-gen "a cozy cabin" --backend qwen --size 768x768 --fmt png --output cabin.png
#   image-gen "a dragon" --backend hunyuan --fmt jpg --output dragon.jpg
# Notes:
#   - Gemini requires GEMINI_API_KEY in your environment.
#   - Qwen/Hunyuan require optional extras: `pip install -e .[qwen]` / `pip install -e .[hunyuan]`.

import argparse
import asyncio
from pathlib import Path

from imagen.backends import get_backend


async def _run_async(args):
    backend = get_backend(args.backend)
    result = await backend.generate_image(
        prompt=args.prompt,
        size=args.size,
        fmt=args.fmt,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
    )
    out_path = Path(args.output or result.filename)
    out_path.write_bytes(result.content)
    print(str(out_path))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Generate an image (no MCP)")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--size", default="1024x1024", help="Size WxH, default 1024x1024")
    parser.add_argument("--fmt", default="png", choices=["png", "jpg", "jpeg", "webp"], help="Image format")
    parser.add_argument("--backend", default=None, help="mock|gemini|qwen|hunyuan|auto (default auto)")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args(argv)
    asyncio.run(_run_async(args))


if __name__ == "__main__":  # pragma: no cover
    main()
