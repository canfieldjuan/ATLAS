#!/usr/bin/env python3
"""Standalone-mode smoke check for extracted_content_pipeline.

Walks ``extracted_content_pipeline/manifest.json``, imports every
Python target under ``EXTRACTED_PIPELINE_STANDALONE=1``, and exits
non-zero on any **real decoupling failure**.

A "real decoupling failure" is an ImportError for a name that starts
with ``extracted_`` or ``atlas_brain`` -- a missing relative-import
target that means the standalone substrate is incomplete. Other
import failures (missing 3rd-party packages like ``httpx`` or
``pydantic`` in a stripped CI env) are reported as warnings but do
not fail the gate -- those are a different concern handled elsewhere.

Manifest-driven (not hardcoded). New entries under ``mappings`` or
``owned`` are automatically covered.

See plans/PR-Audit-PipelineStandaloneSmoke-1.md for the rationale.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "extracted_content_pipeline"
ENV_VAR = "EXTRACTED_PIPELINE_STANDALONE"


def _target_to_module(target: str) -> str:
    """Convert a manifest path target to a Python module name."""
    assert target.endswith(".py"), target
    base = target[: -len(".py")]
    if base.endswith("/__init__"):
        base = base[: -len("/__init__")]
    return base.replace("/", ".")


def _is_decoupling_failure(stderr: str) -> bool:
    """A decoupling failure references an extracted_* or atlas_brain
    module that should have been resolvable but was not."""
    if "extracted_" in stderr and "ModuleNotFoundError" in stderr:
        return True
    if "atlas_brain" in stderr and "ModuleNotFoundError" in stderr:
        return True
    return False


def _load_targets() -> list[str]:
    manifest_path = ROOT / PACKAGE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("mappings", []) + manifest.get("owned", [])
    return sorted(
        e["target"]
        for e in entries
        if e["target"].endswith(".py")
        and "/migrations/" not in e["target"]
        and not e["target"].endswith("/__init__.py")
    )


def main() -> int:
    targets = _load_targets()
    if not targets:
        print(f"FAIL no python targets discovered in {PACKAGE}/manifest.json", file=sys.stderr)
        return 2

    decoupling_failures: list[tuple[str, str]] = []
    env_failures: list[tuple[str, str]] = []
    ok_count = 0

    env = os.environ.copy()
    env[ENV_VAR] = "1"

    for target in targets:
        module = _target_to_module(target)
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            ok_count += 1
            continue
        stderr = result.stderr.strip()
        last_line = stderr.splitlines()[-1] if stderr else "(no stderr)"
        if _is_decoupling_failure(stderr):
            decoupling_failures.append((module, last_line))
        else:
            env_failures.append((module, last_line))

    print(f"=== {PACKAGE} standalone smoke ({ENV_VAR}=1) ===")
    print(f"  imported OK : {ok_count}")
    print(f"  decoupling failures: {len(decoupling_failures)}")
    print(f"  3rd-party env failures: {len(env_failures)}")

    if decoupling_failures:
        print()
        print("REAL DECOUPLING FAILURES (gate-breaking):")
        for module, err in decoupling_failures:
            print(f"  {module}")
            print(f"    -> {err}")

    if env_failures:
        print()
        print("3rd-party env failures (warning only, not gate-breaking):")
        for module, err in env_failures:
            print(f"  {module}: {err}")

    return 1 if decoupling_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
