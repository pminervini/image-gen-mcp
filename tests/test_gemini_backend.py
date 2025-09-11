# -*- coding: utf-8 -*-

import pytest

from imagen.backends.gemini import GeminiBackend


@pytest.mark.asyncio
async def test_gemini_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    backend = GeminiBackend()
    with pytest.raises(RuntimeError) as ei:
        await backend.generate_image("a test prompt")
    assert "API key" in str(ei.value)

