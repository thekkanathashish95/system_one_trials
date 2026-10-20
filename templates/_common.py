"""Shared helpers for the templates: makes `jev` importable and handles --dry-run."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv is optional
    pass

if "--dry-run" in sys.argv:
    os.environ["JEV_DRY_RUN"] = "1"


def pct(p: float) -> str:
    return f"{p * 100:5.1f}%"
