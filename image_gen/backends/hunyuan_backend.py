from __future__ import annotations

import io
import os
from typing import Optional, Tuple

import torch  # type: ignore
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline  # type: ignore

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


def _select_device_and_dtype() -> tuple[str, str]:
    """Pick best local device and dtype string for Hunyuan.

    Returns a tuple of (device, dtype_str) where dtype_str is one of 'bf16', 'fp16', 'fp32'.
    """
    # CUDA, prefer bf16 if supported, else fp16
    if torch.cuda.is_available():
        bf16_ok = False
        try:
            bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        except Exception:
            bf16_ok = False
        return ("cuda", "bf16" if bf16_ok else "fp16")
    # Apple MPS
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return ("mps", "fp16")
    # CPU
    return ("cpu", "fp32")


class HunyuanBackend(ImageBackend):
    """Local HunyuanImage-2.1 inference via upstream pipeline.

    Requirements (not auto-installed):
      - Clone https://github.com/Tencent-Hunyuan/HunyuanImage-2.1 and install deps:
        pip install -r requirements.txt
        pip install flash-attn==2.7.3 --no-build-isolation  # for CUDA
      - Make sure Python can import the package (e.g., `pip install -e .` in the repo).
      - Download checkpoints locally and set env `HUNYUANIMAGE_V2_1_MODEL_ROOT` to the directory
        containing `vae`, `text_encoder`, and `dit` subfolders as per upstream docs.
        Alternatively set `HUNYUAN_MODEL_ROOT` (this backend will map it to the expected env).

    This backend runs fully locally and selects CUDA → MPS → CPU automatically.
    """

    name = "hunyuan"

    def __init__(self, model_name: Optional[str] = None):
        # 'hunyuanimage-v2.1' or 'hunyuanimage-v2.1-distilled'
        self.model_name = model_name or os.getenv("HUNYUAN_MODEL_NAME", "hunyuanimage-v2.1")
        self._pipe = None
        self._device = None
        self._dtype = None

    def _ensure_env(self):
        # Allow users to set a generic root, map it to upstream env var name
        root = os.getenv("HUNYUAN_MODEL_ROOT")
        if root and not os.getenv("HUNYUANIMAGE_V2_1_MODEL_ROOT"):
            os.environ["HUNYUANIMAGE_V2_1_MODEL_ROOT"] = root

    def _ensure_pipe(self):
        if self._pipe is not None:
            return
        self._ensure_env()
        device, dtype_str = _select_device_and_dtype()
        self._device, self._dtype = device, dtype_str

        # Construct pipeline with local weights; dtype/device are strings in this pipeline
        pipe = HunyuanImagePipeline.from_pretrained(
            model_name=self.model_name,
            torch_dtype=dtype_str,
            device=device,
        )

        try:
            pipe = pipe.to(device)
        except Exception:
            # If move fails, fallback to CPU
            device = "cpu"
            pipe = pipe.to(device)
            self._device, self._dtype = device, "fp32"

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

        width, height = _parse_size(size)
        use_reprompt = os.getenv("HUNYUAN_USE_REPROMPT", "false").lower() in ("1", "true", "yes")
        use_refiner = os.getenv("HUNYUAN_USE_REFINER", "false").lower() in ("1", "true", "yes")

        image = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            use_reprompt=use_reprompt,
            use_refiner=use_refiner,
            seed=seed,
        )

        buffer = io.BytesIO()
        fmt_upper = fmt.upper()
        if fmt_upper == "JPG":
            fmt_upper = "JPEG"
        image.save(buffer, format=fmt_upper)
        content = buffer.getvalue()

        fmt_lower = fmt.lower()
        content_type = f"image/{'jpeg' if fmt_lower == 'jpg' else fmt_lower}"
        filename = f"hunyuan_{abs(hash(prompt)) % 1_000_000}.{fmt_lower}"

        return ImageResult(content=content, content_type=content_type, format=fmt_lower, filename=filename)
