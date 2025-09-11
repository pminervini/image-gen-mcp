#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Usage examples (Hunyuan local upstream pipeline):
#   pip install -e .[hunyuan]   # once, plus install upstream 'hyimage' package
#   export HUNYUANIMAGE_V2_1_MODEL_ROOT=/path/to/ckpts
#   PYTHONPATH=. python3 cli/hunyuan-cli.py "a sunny field" --size 1024x1024 --fmt png --output field.png
# Notes:
#   - Runs locally; selects CUDA → MPS → CPU and an efficient dtype.
#   - Accepts --seed, --size, --negative-prompt; see README for setup.

import argparse
import asyncio
from pathlib import Path

from imagen.backends.hunyuan import HunyuanBackend


async def _run_async(args):
    backend = HunyuanBackend()
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
    parser = argparse.ArgumentParser(description="Generate an image with Hunyuan backend")
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
