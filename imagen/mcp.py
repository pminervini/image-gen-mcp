# -*- coding: utf-8 -*-

import argparse
import asyncio
import base64
from typing import Optional

from .backends import get_backend


async def run_stdio():
    try:
        from mcp.server import Server  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
    except Exception as e:  # pragma: no cover - only hit if dependency missing
        raise SystemExit(
            "The 'mcp' package is required for stdio mode. Install with: pip install mcp"
        ) from e

    server = Server("imagen-mcp")

    @server.tool()
    async def generate_image(prompt: str, size: str = "1024x1024", fmt: str = "png", backend: Optional[str] = None) -> dict:
        """Generate an image from a prompt. Returns JSON with base64-encoded image.

        Args:
            prompt: Text prompt for image generation
            size: Image size string like "1024x1024"
            fmt: Output image format (png|jpg|jpeg|webp)
            backend: Which backend to use (gemini|qwen|hunyuan|mock|auto)
        """
        b = get_backend(backend)
        result = await b.generate_image(prompt=prompt, size=size, fmt=fmt)
        return {
            "content_type": result.content_type,
            "format": result.format,
            "filename": result.filename,
            "base64": base64.b64encode(result.content).decode("utf-8"),
        }

    async with stdio_server() as (read, write):
        await server.run(read, write)


def main():
    parser = argparse.ArgumentParser(description="Imagen MCP Server")
    parser.add_argument("--transport", choices=["stdio"], default="stdio")
    args = parser.parse_args()
    if args.transport == "stdio":
        asyncio.run(run_stdio())
    else:  # pragma: no cover
        raise SystemExit("Unsupported transport")


if __name__ == "__main__":  # pragma: no cover
    main()
