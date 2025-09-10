# -*- coding: utf-8 -*-

import io
from typing import Optional, Tuple

import torch  # type: ignore
from diffusers import DiffusionPipeline  # type: ignore

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


def _select_device() -> tuple[str, torch.dtype]:
    """Select the best available torch device and dtype.

    Prefers CUDA, then MPS, then CPU. Uses float16 for GPU/MPS, float32 for CPU.
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16
    # mps may exist on macOS
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps", torch.float16
    return "cpu", torch.float32


class QwenImageBackend(ImageBackend):
    """Text-to-image using Qwen/Qwen-Image via diffusers.

    This backend loads the Hugging Face diffusers pipeline lazily on first use.
    It attempts to use CUDA or MPS if available, otherwise CPU.
    """

    name = "qwen"

    def __init__(self, model_id: str = "Qwen/Qwen-Image"):
        self.model_id = model_id
        self._pipe = None
        self._device = None
        self._dtype = None

    def _ensure_pipe(self):
        if self._pipe is not None:
            return
        device, dtype = _select_device()
        self._device, self._dtype = device, dtype

        # Load pipeline; prefer fp16 on GPU/MPS. Keep CPU in fp32.
        kwargs = {"use_safetensors": True}
        if self._device in ("cuda", "mps"):
            kwargs["torch_dtype"] = dtype

        pipe = DiffusionPipeline.from_pretrained(self.model_id, **kwargs)

        # Move to device and enable common memory optimizations
        pipe = pipe.to(self._device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self._pipe = pipe

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        fmt: str = "png",
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> ImageResult:
        self._ensure_pipe()
        assert self._pipe is not None

        import torch  # type: ignore

        width, height = _parse_size(size)

        # For CUDA we can use a CUDA generator; for MPS, CPU generator is typically safer
        generator = None
        if seed is not None:
            gen_device = "cuda" if self._device == "cuda" else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Run pipeline
        out = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            generator=generator,
        )

        image = out.images[0]
        buffer = io.BytesIO()
        fmt_upper = fmt.upper()
        if fmt_upper == "JPG":
            fmt_upper = "JPEG"
        image.save(buffer, format=fmt_upper)
        content = buffer.getvalue()

        fmt_lower = fmt.lower()
        content_type = f"image/{'jpeg' if fmt_lower == 'jpg' else fmt_lower}"
        filename = f"qwen_{abs(hash(prompt)) % 1_000_000}.{fmt_lower}"

        return ImageResult(content=content, content_type=content_type, format=fmt_lower, filename=filename)
