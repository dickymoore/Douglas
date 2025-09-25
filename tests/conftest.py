"""Pytest configuration for Douglas tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

SRC_ROOT = PROJECT_ROOT / "src"
SRC_ROOT_STR = str(SRC_ROOT)
if SRC_ROOT.exists() and SRC_ROOT_STR not in sys.path:
    sys.path.insert(0, SRC_ROOT_STR)
