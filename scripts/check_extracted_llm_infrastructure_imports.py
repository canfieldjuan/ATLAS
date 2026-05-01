#!/usr/bin/env python3
"""Compatibility wrapper for the shared LLM-infrastructure import check."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [
        sys.executable,
        str(ROOT / "extracted" / "_shared" / "scripts" / "check_extracted_imports.py"),
        "extracted_llm_infrastructure",
    ]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
