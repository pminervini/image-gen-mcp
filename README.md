# Imagen MCP

Simple MCP server for generating images, plus a direct CLI. Supports Google Gemini, Qwen/Qwen-Image (diffusers, local), and Tencent HunyuanImage-2.1 (local via upstream pipeline), with a built‑in mock backend for offline/dev usage.

- MCP server over stdio (tool: `generate_image`)
- CLI utility to generate images without MCP
- Unit tests

Note: The Gemini backend requires a `GEMINI_API_KEY`. The mock backend works without any external services.

## Quick Start

- Install (dev):
  - `pip install -e .[dev]`

- Generate an image via CLI (no MCP):
  - `image-gen "A red square" --backend mock --fmt png --output red.png`
  - `image-gen "A scenic lake at sunrise" --backend qwen --fmt png --output lake.png`
  - `image-gen "A futuristic cityscape" --backend hunyuan --fmt jpg --output city.jpg`

### Backend-Specific CLIs (direct)

- Mock: `image-gen-mock "A red square" --fmt png --output red.png`
- Gemini: `image-gen-gemini "A beach at sunset" --fmt png --output beach.png`
- Qwen (diffusers, local): `image-gen-qwen "a cozy cabin" --fmt png --output cabin.png`
- Hunyuan (local upstream pipeline): `image-gen-hunyuan "a dragon" --fmt jpg --output dragon.jpg`

## MCP Server

Run the MCP server over stdio:

- `image-gen-mcp --transport stdio`

Tools:

- `generate_image(prompt, size="1024x1024", fmt="png", backend=None)` → returns JSON with base64 image and metadata

## Gemini Backend

- Set `GEMINI_API_KEY` in your environment.
- By default, the code selects `gemini` backend automatically if the API key is present; otherwise it uses `mock`.
- The implementation uses the `google-genai` package and generates images via `client.models.generate_content` using the image-capable model `gemini-2.5-flash-image-preview`. If your SDK/model availability differs, you can override the model name when constructing the backend or set `IMAGE_GEN_BACKEND=mock`.

## Qwen Backend (diffusers)

- Uses Hugging Face diffusers to run `Qwen/Qwen-Image` locally.
- Prefers CUDA, then MPS, then CPU.
- Install optional dependencies: `pip install -e .[qwen]`
- Example: `image-gen "a cozy cabin in the woods" --backend qwen --fmt png --output cabin.png`

## Hunyuan Backend (local upstream pipeline)

- Uses the official Hunyuan pipeline from the upstream repository, running locally (no remote inference).
- Setup steps (summary; see upstream docs for details):
  - Clone repo: `git clone https://github.com/Tencent-Hunyuan/HunyuanImage-2.1`
  - Install deps in this project: `pip install -e .[hunyuan]`
  - Install upstream repo's own extras if needed; for CUDA you may also install `flash-attn==2.7.3 --no-build-isolation`
  - Make `hyimage` importable (e.g., `pip install -e .` inside the repo)
  - Download checkpoints and set `HUNYUANIMAGE_V2_1_MODEL_ROOT=/path/to/ckpts` (or set `HUNYUAN_MODEL_ROOT` which this project maps to the upstream env var)
- The backend auto-selects device: CUDA → MPS → CPU, and uses an efficient dtype (bf16/fp16/fp32) based on hardware.
- Optional envs: `HUNYUAN_MODEL_NAME` (`hunyuanimage-v2.1` or `hunyuanimage-v2.1-distilled`), `HUNYUAN_USE_REPROMPT=1`, `HUNYUAN_USE_REFINER=1`.
- Example: `image-gen "a dragon flying over mountains" --backend hunyuan --fmt jpg --output dragon.jpg`

## Development

- Run tests: `pytest`
- Lint/format: not configured; keep changes minimal and consistent.

## Project Layout

- `imagen/` — package with MCP server and backends
- `cli/` — CLI (`main-cli.py`)
- `tests/` — unit tests

## Notes

- The mock backend uses Pillow to synthesize an image and works offline.
