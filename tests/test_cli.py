# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import importlib.util


def _load_main_cli():
    # Load cli/main-cli.py directly as a module
    root = Path(__file__).resolve().parents[1]
    main_cli_path = root / "cli" / "main-cli.py"
    spec = importlib.util.spec_from_file_location("cli_main", str(main_cli_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load main CLI from {main_cli_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


gen_image = _load_main_cli()


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
