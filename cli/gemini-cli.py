#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Usage examples (Gemini backend):
#   export GEMINI_API_KEY=... && image-gen-gemini "A beach at sunset" --fmt png --output beach.png
#   python -m cli.gemini_cli "A robot" --size 512x512 --fmt jpg --output robot.jpg
# Notes:
#   - Requires google-genai and GEMINI_API_KEY set in the environment.

import argparse
import asyncio
import sys
from pathlib import Path

from imagen.backends.gemini import GeminiBackend


async def _run_async(args):
    backend = GeminiBackend(api_key=getattr(args, "api_key", None))
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


from typing import Optional, List


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Generate an image with Gemini backend")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--size", default="1024x1024", help="Size WxH, default 1024x1024")
    parser.add_argument("--fmt", default="png", choices=["png", "jpg", "jpeg", "webp"], help="Image format")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--api-key", default=None, help="Gemini API key (overrides env)")
    args = parser.parse_args(argv)
    asyncio.run(_run_async(args))


if __name__ == "__main__":  # pragma: no cover
    main()
