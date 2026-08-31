#!/usr/bin/env python3
"""Compatibility launcher for the reorganized SafeVision Web API."""

import importlib.util
from pathlib import Path


_APP_PATH = Path(__file__).resolve().parent / "SafeVision Web API" / "app.py"
_SPEC = importlib.util.spec_from_file_location("safevision_web_api_app", _APP_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load SafeVision Web API from {_APP_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

app = _MODULE.app
api_instance = _MODULE.api_instance
API_CONFIG = _MODULE.API_CONFIG
main = _MODULE.main

if __name__ == "__main__":
    main()
