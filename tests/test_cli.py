# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import cli.main_cli as gen_image


def test_cli_mock(tmp_path: Path, monkeypatch):
    out_file = tmp_path / "out.png"
    argv = [
        "A test prompt",
        "--backend",
        "mock",
        "--fmt",
        "png",
        "--output",
        str(out_file),
    ]
    gen_image.main(argv)
    assert out_file.exists()
    assert out_file.stat().st_size > 100
