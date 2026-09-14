from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def plugin():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "comfyui_auk_t8_tests",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
