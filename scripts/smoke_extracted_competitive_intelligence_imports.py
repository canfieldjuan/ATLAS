#!/usr/bin/env python3
"""Default-mode import smoke for extracted_competitive_intelligence (Phase 1).

Manifest-driven: walks ``extracted_competitive_intelligence/manifest.json``
(``mappings`` + ``owned``) at runtime and imports every Python target.
New mirrors get auto-coverage -- no hardcoded MODULES list to forget
to update when a new file enters the manifest.

The scaffold is a verbatim snapshot of atlas_brain sources, so an
ImportError referencing ``extracted_*`` or ``atlas_brain`` means the
manifest pulled in a module whose imports are not resolvable in the
scaffold's package layout. Phase 1 satisfies absolute ``atlas_brain.*``
imports by relying on the parent atlas_brain package being importable
(the scaffold sits alongside it on ``sys.path``); Phase 2 replaces
those absolute imports with seams that let the scaffold run without
atlas_brain on the path.

Failure classification:
- ``ModuleNotFoundError`` referencing ``extracted_*`` or ``atlas_brain``
  is a gate-breaking decoupling failure (the drift this smoke is
  meant to catch).
- Any other exception at import time -- missing 3rd-party packages,
  Pydantic settings validation, etc. -- is an env failure: reported
  as a warning but does not break the gate.

The narrower gate keeps the smoke focused on its purpose. Other
import-time errors are caught by complementary checks
(``check_extracted_competitive_intelligence_imports.py``,
``forbid_atlas_reasoning_imports.py``).

See plans/PR-Audit-ManifestDrivenSmokes-1.md for rationale.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "extracted_competitive_intelligence"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _target_to_module(target: str) -> str:
    assert target.endswith(".py"), target
    base = target[: -len(".py")]
    if base.endswith("/__init__"):
        base = base[: -len("/__init__")]
    return base.replace("/", ".")


def _is_decoupling_failure(exc: BaseException) -> bool:
    if not isinstance(exc, ModuleNotFoundError):
        return False
    name = getattr(exc, "name", "") or ""
    return name.startswith("extracted_") or name == "atlas_brain" or name.startswith("atlas_brain.")


def _load_modules() -> list[str]:
    manifest = json.loads((ROOT / PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("mappings", []) + manifest.get("owned", [])
    return sorted({
        _target_to_module(e["target"])
        for e in entries
        if e["target"].endswith(".py")
        and "/migrations/" not in e["target"]
        and not e["target"].endswith("/__init__.py")
    })


def main() -> int:
    modules = _load_modules()
    if not modules:
        print(f"FAIL no python targets discovered in {PACKAGE}/manifest.json", file=sys.stderr)
        return 2

    decoupling_failures: list[tuple[str, str]] = []
    env_failures: list[tuple[str, str]] = []
    ok = 0

    for module in modules:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if _is_decoupling_failure(exc):
                decoupling_failures.append((module, msg))
            else:
                env_failures.append((module, msg))
        except Exception as exc:
            env_failures.append((module, f"{type(exc).__name__}: {exc}"))
        else:
            ok += 1

    print(f"=== {PACKAGE} default-mode smoke ===")
    print(f"  imported OK : {ok}")
    print(f"  decoupling failures: {len(decoupling_failures)}")
    print(f"  env failures: {len(env_failures)}")

    if decoupling_failures:
        print()
        print("REAL DECOUPLING FAILURES (gate-breaking):")
        for module, err in decoupling_failures:
            print(f"  {module}: {err}")

    if env_failures:
        print()
        print("Env failures (warning only -- not gate-breaking):")
        for module, err in env_failures:
            print(f"  {module}: {err}")

    if decoupling_failures:
        return 1

    print()
    print("Import smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
