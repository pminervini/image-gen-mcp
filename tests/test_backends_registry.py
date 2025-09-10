# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest
from imagen.backends import get_backend


def test_get_backend_qwen_instance():
    pytest.importorskip("diffusers")
    pytest.importorskip("torch")
    b = get_backend("qwen")
    assert getattr(b, "name", "") == "qwen"


def test_get_backend_hunyuan_instance():
    pytest.importorskip("hyimage")
    pytest.importorskip("torch")
    b = get_backend("hunyuan")
    assert getattr(b, "name", "") == "hunyuan"
