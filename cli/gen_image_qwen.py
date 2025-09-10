#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Usage examples (Qwen via diffusers, local):
#   pip install -e .[qwen]   # once
#   image-gen-qwen "a cozy cabin" --size 768x768 --fmt png --output cabin.png
#   python -m cli.gen_image_qwen "A dragon" --seed 42 --fmt jpg --output dragon.jpg
# Notes:
#   - Prefers CUDA → MPS → CPU automatically.
#   - Installs diffusers/torch and related extras via the `[qwen]` extra.

import argparse
import asyncio
import sys
from pathlib import Path


async def _run_async(args):
    try:
        from imagen.backends.qwen import QwenImageBackend
    except Exception as e:  # pragma: no cover - import-time optional deps
        raise SystemExit(
            "Qwen backend requires optional deps. Install with: pip install -e .[qwen]"
        ) from e

    backend = QwenImageBackend()
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
    parser = argparse.ArgumentParser(description="Generate an image with Qwen (diffusers) backend")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--size", default="1024x1024", help="Size WxH, default 1024x1024")
    parser.add_argument("--fmt", default="png", choices=["png", "jpg", "jpeg", "webp"], help="Image format")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args(argv)
    asyncio.run(_run_async(args))


if __name__ == "__main__":  # pragma: no cover
    main()
