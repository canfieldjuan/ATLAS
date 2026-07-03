#!/usr/bin/env python3
"""Thin CLI wrapper for the Reddit fit evaluation harness.

All logic lives in atlas_reddit.fit_eval; this wrapper only makes the
harness runnable from the repo root without installing the package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_reddit.fit_eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
